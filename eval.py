#!/usr/bin/env python3
import os
import time
import subprocess
import csv
import sqlite3
import traceback
import signal
import threading
import queue
import json
import shutil
import contextlib
import concurrent.futures
from pathlib import Path

import docker

# set the number of concurrent runs
NUM_CPUS = os.cpu_count()

# If container logs should be shown
SHOW_CONTAINER_LOGS = False

# Timeout for the fuzzers in seconds
TIMEOUT = 60 * 60

# Time interval in which to check the results of a fuzzer
CHECK_INTERVAL = 5

# The path where eval data is stored outside of the docker container
HOST_TMP_PATH = Path(".").resolve()/"tmp/"

# The location where the eval data is mapped to inside the docker container
IN_DOCKER_WORKDIR = "/workdir/"

TRIGGERED_STR = b"Triggered!\r\n"

# The programs that can be evaluated
PROGRAMS = {
    # "objdump": {
    #     "include": "",
    #     "path": "binutil/binutil/",
    #     "args": "--dwarf-check -C -g -f -dwarf -x @@",
    # },
    # "re2": {
    #     "compile_args": [
    #         # {'val': "-v", 'action': None},
    #         # {'val': "-static", 'action': None},
    #         # {'val': "-std=c++11", 'action': None},
    #         {'val': "-lpthread", 'action': None},
    #         # {'val': "samples/re2/re2_fuzzer.cc", 'action': "prefix_workdir"},
    #         {'val': "-I", 'action': None},
    #         {'val': "samples/re2-code/", 'action': "prefix_workdir"},
    #         # {'val': "-lc++", 'action': None},
    #         # {'val': "-lstdc++", 'action': None},
    #         # {'val': "-D_GLIBCXX_USE_CXX11_ABI=0", 'action': None},
    #     ],
    #     "path": "samples/re2/",
    #     "args": "@@",
    # },
    "guetzli": {
        "compile_args": [
            # {'val': "-v", 'action': None},
            # {'val': "-static", 'action': None},
            # {'val': "-std=c++11", 'action': None},
            # {'val': "-lpthread", 'action': None},
            # {'val': "samples/re2/re2_fuzzer.cc", 'action': "prefix_workdir"},
            # {'val': "-I", 'action': None},
            # {'val': "samples/re2-code/", 'action': "prefix_workdir"},
            # {'val': "-lc++", 'action': None},
            # {'val': "-lstdc++", 'action': None},
            # {'val': "-D_GLIBCXX_USE_CXX11_ABI=0", 'action': None},
            # {'val': "samples/guetzli/fuzz_target.bc", 'action': "prefix_workdir"},
        ],
        "orig_bin": str(Path("tmp/samples/guetzli/fuzz_target").absolute()),
        "path": "samples/guetzli/",
        "seeds": "samples/guetzli_harness/seeds/",
        "args": "@@",
    },
}

# Indicates if the evaluation should continue, is mainly used to shut down 
# after a keyboard interrupt by the user.
# Global variable that is only written in the sigint_handler, as such it is safe
# to use in a read only fashion by the threads.
should_run = True

# Handler for a keyboard interrupt only sets `should_run` to False.
def sigint_handler(signum, frame):
    global should_run
    print("Got stop signal, stopping", signum)
    should_run = False

# A helper function to reduce load on the database and reduce typing overhead
def connection(f):
    def wrapper(self, *args, **kwargs):
        if self.conn is None:
            return
        res = f(self, self.conn.cursor(), *args, **kwargs)
        if self._time_last_commit + 5 > time.time():
            self.conn.commit()
            self._time_last_commit = time.time()
        return res
    return wrapper

# A class to store information into a sqlite database. This ecpects sole access
# to the database.
class Stats():

    def __init__(self, db_path):
        super().__init__()
        if db_path is None:
            print(f"Didn't get db_path env, not writing history.")
            self.conn = None
            return
        db_path = Path(db_path)
        print(f"Writing history to: {db_path}")
        if db_path.is_file():
            print(f"DB exists, deleting: {db_path}")
            db_path.unlink()
        # As we have sole access, we just drop all precaution needed for
        # multi-party access. This removes overhead.
        self.conn = sqlite3.connect(str(db_path), isolation_level="Exclusive")
        # Initialize the tables
        self._init_tables()
        c = self.conn.cursor()
        # Same as above, this reduces some overhead.
        c.execute('PRAGMA synchronous = 0')
        c.execute('PRAGMA journal_mode = OFF') 
        # Record when we started
        self._start_time = time.time()
        # Record the last time we committed
        self._time_last_commit = time.time()

    def _init_tables(self):
        c = self.conn.cursor()

        c.execute('''
        CREATE TABLE runs (
            prog,
            mutation_id,
            fuzzer,
            workdir,
            prog_bc,
            compile_args,
            args,
            seeds,
            orig_bin,
            mut_additional_info,
            mut_column,
            mut_directory,
            mut_file_path,
            mut_line,
            mut_type
        )''')

        c.execute('''
        CREATE TABLE executed_runs (
            prog,
            mutation_id,
            fuzzer,
            covered_file_seen,
            total_time
        )''')

        c.execute('''
        CREATE TABLE aflpp_runs (
            prog,
            mutation_id,
            fuzzer,
            time,
            cycles_done,
            cur_path,
            paths_total,
            pending_total,
            pending_favs,
            map_size,
            unique_crashes,
            unique_hangs,
            max_depth,
            execs_per_sec,
            totals_execs
        )''')

        c.execute('''
        CREATE TABLE crashing_inputs (
            prog,
            mutation_id,
            fuzzer,
            time_found,
            stage,
            path,
            crashing_input,
            orig_return_code,
            mut_return_code,
            orig_cmd,
            mut_cmd,
            orig_stdout,
            mut_stdout,
            orig_stderr,
            mut_stderr,
            num_triggered
        )''')

        c.execute('''
        CREATE TABLE run_crashed (
            prog,
            mutation_id,
            fuzzer,
            crash_trace
        )''')

        self.conn.commit()

    def commit(self):
        self.conn.commit()

    @connection
    def new_run(self, c, run_data):
        c.execute('INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_data['prog'],
                run_data['mutation_id'],
                run_data['fuzzer'],
                str(run_data['workdir']),
                str(run_data['prog_bc']),
                run_data['compile_args'],
                run_data['args'],
                str(run_data['seeds']),
                run_data['orig_bin'],
                json.dumps(run_data['mutation_data']['additionalInfo']),
                run_data['mutation_data']['column'],
                run_data['mutation_data']['directory'],
                run_data['mutation_data']['filePath'],
                run_data['mutation_data']['line'],
                run_data['mutation_data']['type'],
            )
        )
        self.conn.commit()

    @connection
    def new_aflpp_run(self, c, plot_data, prog, mutation_id, fuzzer, cf_seen, total_time):
        c.execute('INSERT INTO executed_runs VALUES (?, ?, ?, ?, ?)',
            (
                prog,
                mutation_id,
                fuzzer,
                cf_seen,
                total_time,
            )
        )
        start_time = None
        for row in plot_data:
            cur_time = int(row['# unix_time'].strip())
            if start_time is None:
                start_time = cur_time
            cur_time -= start_time
            c.execute('INSERT INTO aflpp_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    prog,
                    mutation_id,
                    fuzzer,
                    cur_time,
                    row[' cycles_done'].strip(),
                    row[' cur_path'].strip(),
                    row[' paths_total'].strip(),
                    row[' pending_total'].strip(),
                    row[' pending_favs'].strip(),
                    row[' map_size'].strip(),
                    row[' unique_crashes'].strip(),
                    row[' unique_hangs'].strip(),
                    row[' max_depth'].strip(),
                    row[' execs_per_sec'].strip(),
                    row[None][0].strip(),
                )
            )
        self.conn.commit()

    @connection
    def new_crashing_inputs(self, c, crashing_inputs, prog, mutation_id, fuzzer):
        for path, data in crashing_inputs.items():
            c.execute('INSERT INTO crashing_inputs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    prog,
                    mutation_id,
                    fuzzer,
                    data['time_found'],
                    data['stage'],
                    path,
                    data['data'],
                    data['orig_returncode'],
                    data['mut_returncode'],
                    ' '.join((str(v) for v in data['orig_cmd'])),
                    ' '.join((str(v) for v in data['mut_cmd'])),
                    data['orig_res'][0],
                    data['mut_res'][0],
                    data['orig_res'][1],
                    data['mut_res'][1],
                    data['num_triggered']
                )
            )
        self.conn.commit()

    @connection
    def run_crashed(self, c, prog, mutation_id, fuzzer, trace):
        c.execute('INSERT INTO run_crashed VALUES (?, ?, ?, ?)',
            (
                prog,
                mutation_id,
                fuzzer,
                trace,
            )
        )
        self.conn.commit()

class DockerLogStreamer(threading.Thread):
    def __init__(self, q, container, *args, **kwargs):
        self.q = q
        self.container = container
        super().__init__(*args, **kwargs)

    def run(self):
        global should_run
        for line in self.container.logs(stream=True):
            line = line.decode()
            if SHOW_CONTAINER_LOGS:
                print(line.rstrip())
            self.q.put(line)
        self.q.put(None)

@contextlib.contextmanager
def start_testing_container(core_to_use, trigger_file):
    # get access to the docker client to start the container
    docker_client = docker.from_env()

    # build testing image
    proc = subprocess.run([
            "docker", "build",
            "-t", "mutator_testing",
            "-f", "eval/Dockerfile.testing",
            "."
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print("Could not build testing image.", proc)
        raise ValueError(proc)

    # Start and run the container
    container = docker_client.containers.run(
        "mutator_testing", # the image
        ["sleep", str(TIMEOUT * 2 + 120)], # the arguments
        init=True,
        ipc_mode="host",
        auto_remove=True,
        environment={
            'LD_LIBRARY_PATH': "/workdir/lib/",
            'TRIGGERED_FILE': str(trigger_file),
        },
        volumes={str(HOST_TMP_PATH): {'bind': str(IN_DOCKER_WORKDIR),
                                      'mode': 'ro'}},
        working_dir=str(IN_DOCKER_WORKDIR),
        cpuset_cpus=str(core_to_use),
        mem_limit="1g",
        detach=True
    )
    yield container
    container.stop()
        
def run_exec_in_testing_container(container, cmd):
    sub_cmd = ["docker", "exec", "-it",
        container.name,
        *cmd]
    return subprocess.run(sub_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

class CoveredFile:
    def __init__(self, workdir, start_time) -> None:
        super().__init__()
        self.found = None
        self.path = Path(workdir)/"covered"
        self.start_time = start_time

        if self.path.is_file():
            self.path.unlink()

    def check(self):
        if self.found is None and self.path.is_file():
            self.found = time.time() - self.start_time

# returns true if a crashing input is found that only triggers for the
# mutated binary
def check_crashing_inputs(testing_container, crashing_inputs, crash_dir,
                          orig_bin, mut_bin, args, start_time, covered, stage):
    if not crash_dir.is_dir():
        return False

    for path in crash_dir.iterdir():
        if path.is_file() and path.name != "README.txt":
            if str(path) not in crashing_inputs:
                # Run input on original binary
                orig_cmd = ["/run_bin.sh", orig_bin]
                orig_cmd += args.replace("@@", str(path)).split(" ")
                proc = run_exec_in_testing_container(testing_container, orig_cmd)
                orig_res = (proc.stdout, proc.stderr)
                orig_returncode = proc.returncode

                # Run input on mutated binary
                mut_cmd = ["/run_bin.sh", mut_bin]
                mut_cmd += args.replace("@@", str(path)).split(" ")
                proc = run_exec_in_testing_container(testing_container, mut_cmd)
                mut_res = (proc.stdout, proc.stderr)

                num_triggered = len(mut_res[0].split(TRIGGERED_STR)) - 1
                num_triggered += len(mut_res[1].split(TRIGGERED_STR)) - 1
                mut_res = (
                    mut_res[0].replace(TRIGGERED_STR, b""),
                    mut_res[1].replace(TRIGGERED_STR, b"")
                )
                mut_returncode = proc.returncode

                covered.check()

                try:
                    crash_file_data = path.read_bytes()
                except Exception as exc:
                    crash_file_data = "{}".format(exc)

                crashing_inputs[str(path)] = {
                    'time_found': time.time() - start_time,
                    'stage': stage,
                    'data': crash_file_data,
                    'orig_returncode': orig_returncode,
                    'mut_returncode': mut_returncode,
                    'orig_cmd': orig_cmd,
                    'mut_cmd': mut_cmd,
                    'orig_res': orig_res,
                    'mut_res': mut_res,
                    'num_triggered': num_triggered,
                }

                if (orig_returncode != mut_returncode or orig_res != mut_res):
                    return True
    return False


# Eval function for the afl plus plus fuzzer, compiles the mutated program
# and fuzzes it. Finally various eval data is returned
def aflpp_base_eval(run_data, docker_image, executable):
    global should_run
    # extract used values
    workdir = run_data['workdir']
    prog_bc = IN_DOCKER_WORKDIR/run_data['prog_bc'].relative_to(HOST_TMP_PATH)
    compile_args = run_data['compile_args']
    args = run_data['args']
    seeds = run_data['seeds']
    orig_bin = IN_DOCKER_WORKDIR/Path(run_data['orig_bin']).relative_to(HOST_TMP_PATH)
    core_to_use = run_data['used_core']
    docker_mut_bin = Path(workdir)/"testing"
    docker_mut_bin.parent.mkdir(parents=True, exist_ok=True)

    # get start time for the eval
    start_time = time.time()

    # get path for covered file and rm the file if it exists
    covered = CoveredFile(workdir, start_time)

    # start testing container
    with start_testing_container(core_to_use, covered.path) as testing_container:

        # compile the compare version of the mutated binary
        compile_mut_bin_res = run_exec_in_testing_container(testing_container,
            [
                "/usr/bin/clang++-11",
                "-v",
                "-o", str(docker_mut_bin),
                "/workdir/lib/libdynamiclibrary.so",
                str(prog_bc)
            ]
        )
        if compile_mut_bin_res.returncode != 0:
            raise ValueError(compile_mut_bin_res)

        # set up data for crashing inputs
        crashing_inputs = {}
        crash_dir = workdir/"output"/"crashes"
        # check if seeds are already crashing
        checked_seeds = {}
        # do an initial check to see if the seed files are already crashing
        if check_crashing_inputs(testing_container, checked_seeds, seeds,
                                 orig_bin, docker_mut_bin, args, start_time,
                                 covered, "initial"):
            return {
                'total_time': time.time() - start_time,
                'covered_file_seen': covered.found,
                'plot_data': [],
                'crashing_inputs': checked_seeds,
            }
        # get access to the docker client to start the container
        docker_client = docker.from_env()
        # Start and run the container
        container = docker_client.containers.run(
            docker_image, # the image
            [
                executable,
                str(compile_args),
                str(prog_bc),
                str(IN_DOCKER_WORKDIR/seeds),
                str(args)
            ], # the arguments
            environment={
                'TRIGGERED_OUTPUT': str(""),
                'TRIGGERED_FILE': str(workdir/'covered'),
            },
            init=True,
            cpuset_cpus=str(core_to_use),
            ipc_mode="host",
            auto_remove=True,
            volumes={str(HOST_TMP_PATH): {'bind': str(IN_DOCKER_WORKDIR),
                                          'mode': 'ro'}},
            working_dir=str(workdir),
            mem_limit="1g",
            detach=True
        )

        logs_queue = queue.Queue()
        DockerLogStreamer(logs_queue, container).start()

        while time.time() < start_time + TIMEOUT and should_run:
            # check if the process stopped, this should only happen in an
            # error case
            try:
                container.reload()
            except docker.errors.NotFound:
                # container is dead stop waiting
                break
            if container.status not in ["running", "created"]:
                break

            # Check if covered file is seen
            covered.check()

            # Check if a crashing input has already been found
            if check_crashing_inputs(testing_container, crashing_inputs,
                                     crash_dir, orig_bin, docker_mut_bin, args,
                                     start_time, covered, "runtime"):
                break

            # Sleep so we only check sometimes and do not busy loop 
            time.sleep(CHECK_INTERVAL)

        # Check if container is still running
        try:
            container.reload()
            if container.status in ["running", "created"]:

                # Send sigint to the process
                container.kill(2)

                # Wait up to 10 seconds then send sigkill
                container.stop()
        except docker.errors.NotFound:
            # container is dead just continue maybe it worked
            pass
        
        all_logs = []
        while True:
            line = logs_queue.get()
            if line == None:
                break
            all_logs.append(line)

        try:
                
            # Get the final stats and report them
            with open(workdir/"output"/"plot_data") as csvfile:
                plot_data = list(csv.DictReader(csvfile))
                # only get last row, to reduce memory usage
                try:
                    plot_data = [plot_data[-2]]
                except IndexError:
                    plot_data = [plot_data[-1]]

            # Also collect all crashing outputs
            check_crashing_inputs(testing_container, crashing_inputs, crash_dir,
                                  orig_bin, docker_mut_bin, args, start_time,
                                  covered, "final")
                
            return {
                'total_time': time.time() - start_time,
                'covered_file_seen': covered.found,
                'plot_data': plot_data,
                'crashing_inputs': crashing_inputs,
            }

        except Exception as exc:
            raise ValueError(''.join(all_logs)) from exc

def aflpp_eval(run_data):
    return aflpp_base_eval(
        run_data, "mutator_aflpp", "/home/eval/start_aflpp.sh")

def aflppasan_eval(run_data):
    return aflpp_base_eval(
        run_data, "mutator_aflppasan", "/home/eval/start_aflppasan.sh")

def aflppubsan_eval(run_data):
    return aflpp_base_eval(
        run_data, "mutator_aflppubsan", "/home/eval/start_aflppubsan.sh")

def aflppmsan_eval(run_data):
    return aflpp_base_eval(
        run_data, "mutator_aflppmsan", "/home/eval/start_aflppmsan.sh")

def build_compile_args(args, workdir):
    res = ""
    for arg in args:
        if arg['action'] is None:
            res += arg['val'] + " "
        elif arg['action'] == 'prefix_workdir':
            res += str(Path(workdir)/arg['val']) + " "
        else:
            raise ValueError("Unknown action: {}", arg)
    return res

# The fuzzers that can be evaluated, value is the function used to start an
# evaluation run
# A evaluation function should do following steps:
    # execute the fuzzer while monitoring if the crash was found
    # this includes running the crashing input in the uninstrumented
    # binary to see if it still crashes, as well as running it in the
    # original unmutated binary to see that it does not crash
    # once the crash is found update the stats (executions, time) but
    # also store the crashing input and path to the corresponding
    # mutated binary
FUZZERS = {
    "aflpp": aflpp_eval,
    "aflppasan": aflppasan_eval,
    "aflppubsan": aflppubsan_eval,
    "aflppmsan": aflppmsan_eval,
}

# Generator that first collects all possible runs and adds them to stats.
# Then yields all information needed to start a eval run
def get_next_run(stats):
    all_runs = []
    # For all programs that can be done by our evaluation
    for prog, prog_info in PROGRAMS.items():
        # Get all mutations that are possible with that program
        mutations = list((HOST_TMP_PATH/prog_info['path']/"mutations").glob("*.mut.bc"))
        # get additional info on mutations
        mut_data_path = list(Path(HOST_TMP_PATH/prog_info['path'])
                                .glob('*.ll.mutationlocations'))
        assert(len(mut_data_path) == 1)
        mut_data_path = mut_data_path[0]
        with open(mut_data_path, 'rt') as f:
            mutation_data = json.load(f)
        # Go through the mutations
        for mutation in mutations:
            # For each fuzzer gather all information needed to start a eval run
            for fuzzer, eval_func in FUZZERS.items():
                # Get the mutation id from the file path
                mutation_id = str(mutation).split(".")[-3]
                # Get the working directory based on program and mutation id and
                # fuzzer, which is a unique path for each run
                workdir = Path("/dev/shm/mutator/")/prog/mutation_id/fuzzer
                # Get the bc file that should be fuzzed (and probably
                # instrumented).
                prog_bc = mutation
                # Get the path to the file that should be included during compilation
                compile_args = build_compile_args(prog_info['compile_args'], IN_DOCKER_WORKDIR)
                # Arguments on how to execute the binary
                args = prog_info['args']
                # Prepare seeds
                seeds = Path(prog_info['seeds'])
                # Original binary to check if crashing inputs also crash on
                # unmodified version
                orig_bin = prog_info['orig_bin']
                # gather all info
                run_data = {
                    'fuzzer': fuzzer,
                    'eval_func': eval_func,
                    'workdir': workdir,
                    'prog_bc': prog_bc,
                    'compile_args': compile_args,
                    'args': args,
                    'prog': prog,
                    'seeds': seeds,
                    'orig_bin': orig_bin,
                    'mutation_id': mutation_id,
                    'mutation_data': mutation_data[int(mutation_id)],
                }
                # Add that run to the database, so that we know it is possible
                stats.new_run(run_data)
                # Build our list of runs
                all_runs.append(run_data)

    # Yield the individual eval runs
    for run in all_runs:
        yield run

# Helper function to wait for the next eval run to complete.
# Also updates the stats and which cores are currently used.
# If `break_after_one` is true, return after a single run finishes.
def wait_for_runs(stats, runs, cores_in_use, break_after_one):
    # wait for the futures to complete
    for future in concurrent.futures.as_completed(runs):
        # get the associated data for that future
        data = runs[future]
        # we are not interested in this run anymore
        del runs[future]
        try:
            # if there was no exception get the data
            run_result = future.result()
        except Exception:
            # if there was an exception print it
            trace = traceback.format_exc()
            stats.run_crashed(data['prog'], data['mutation_id'], data['fuzzer'],
                              trace)
            # print('='*50,
            #     '\n%r generated an exception: %s\n' %
            #         (prog_bc, trace), # exc
            #     '='*50)
        else:
            # if successful log the run
            # print("success")
            stats.new_aflpp_run(
                run_result['plot_data'],
                data['prog'],
                data['mutation_id'],
                data['fuzzer'],
                run_result['covered_file_seen'],
                run_result['total_time'])
            stats.new_crashing_inputs(
                run_result['crashing_inputs'], data['prog'],
                data['mutation_id'], data['fuzzer'])
        # Set the core for this run to unused
        cores_in_use[data['used_core']] = False
        # Delete the working directory as it is not needed anymore
        workdir = Path(data['workdir'])
        if workdir.is_dir():
            shutil.rmtree(workdir)
        # If we only wanted to wait for one run, break here to return
        if break_after_one:
            break

def main():
    global should_run
    # prepare environment
    base_shm_dir = Path("/dev/shm/mutator")
    base_shm_dir.mkdir(parents=True, exist_ok=True)

    # set the number of allowed files, this is needed for larger numbers of
    # concurrent runs, this doesn't work here ..., needs to be done in shell
    # os.system("ulimit -n 50000")
    # also set number of available shared memory
    # sudo bash -c "echo 134217728 > /proc/sys/kernel/shmmni"
    # also set core pattern
    # sudo bash -c "echo core >/proc/sys/kernel/core_pattern"

    # set signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, sigint_handler)
    # Initialize the stats object
    stats = Stats("/dev/shm/mutator/stats.db")

    # build the fuzzer docker images
    proc = subprocess.run([
        "docker", "build",
        "-t", "mutator_testing",
        "-f", "eval/Dockerfile.testing",
        "."])
    if proc.returncode != 0:
        print("Could not build testing image.", proc)
        exit(1)

    proc = subprocess.run([
        "docker", "build",
        "-t", "mutator_aflpp",
        "-f", "eval/Dockerfile.aflpp",
        "."])
    if proc.returncode != 0:
        print("Could not build aflpp image.", proc)
        exit(1)

    proc = subprocess.run([
        "docker", "build",
        "-t", "mutator_aflppasan",
        "-f", "eval/Dockerfile.aflppasan",
        "."])
    if proc.returncode != 0:
        print("Could not build aflppasan image.", proc)
        exit(1)

    proc = subprocess.run([
        "docker", "build",
        "-t", "mutator_aflppubsan",
        "-f", "eval/Dockerfile.aflppubsan",
        "."])
    if proc.returncode != 0:
        print("Could not build aflppubsan image.", proc)
        exit(1)

    proc = subprocess.run([
        "docker", "build",
        "-t", "mutator_aflppmsan",
        "-f", "eval/Dockerfile.aflppmsan",
        "."])
    if proc.returncode != 0:
        print("Could not build aflppmsan image.", proc)
        exit(1)

    # for each mutation and for each fuzzer do a run
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
        # keep a list of all runs
        runs = {}
        # start time
        start_time = time.time()
        # Keep a list of which cores can be used
        cores_in_use = [False]*NUM_CPUS
        # Get each run
        for ii, run_data in enumerate(get_next_run(stats)):
            print("" + str(ii) + " @ " + str(int((time.time() - start_time)//60)),
                  end=', ', flush=True)
            # If we should stop, do so now to not create any new run.
            if not should_run:
                break
            # Get a free core, this will throw an exception if there is none
            used_core = cores_in_use.index(False)
            # Set the core as used
            cores_in_use[used_core] = True
            # Unpack the run_data
            # Add the run to the executor, this starts execution of the eval
            # function. Also associate some information with that run, this
            # is later used to record needed information.
            run_data['used_core'] = used_core
            runs[executor.submit(run_data['eval_func'], run_data)] = run_data
            # Do not wait for a run, if we can start more runs do so first and
            # only wait when all cores are in use.
            if False in cores_in_use:
                continue
            # Wait for a single run to complete, after which we can start another
            wait_for_runs(stats, runs, cores_in_use, True)
        # Wait for all remaining active runs
        print('waiting for the rest')
        wait_for_runs(stats, runs, cores_in_use, False)


def parse_afl_paths(paths):
    if paths is None:
        return []
    paths = paths.split('/////')
    paths_elements = []
    for path in paths:
        elements = {}
        split_path = path.split(',')
        elements['path'] = split_path[0]
        for elem in split_path[1:]:
            parts = elem.split(":")
            if len(parts) == 2:
                elements[parts[0]] = parts[1]
            else:
                raise ValueError("Unexpected afl path format: ", path)
        paths_elements.append(elements)
    return paths_elements

def header():
    import altair as alt

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mutation testing eval</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>

        <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
    <style>
    table {{
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        padding: 7px;
        border: none;
    }}
        tr:nth-child(even){{background-color: lightgray}}
    th {{
    }}
    </style>
    </head>
    <body>\n""".format(
        vega_version=alt.VEGA_VERSION,
        vegalite_version=alt.VEGALITE_VERSION,
        vegaembed_version=alt.VEGAEMBED_VERSION,
    )

def runtime_stats(con):
    import pandas as pd
    stats = pd.read_sql_query("SELECT * from run_results_by_fuzzer", con)
    print(stats)
    res = "<h2>Runtime Stats</h2>"
    res += stats.to_html()
    return res

def mut_stats(con):
    import pandas as pd
    stats = pd.read_sql_query("SELECT * from run_results_by_mut_type", con)
    res = "<h2>Mutation Stats</h2>"
    res += stats.to_html()
    return res

def add_datapoint(crash_ctr, data, mut_type, fuzzer, prog, total_num_muts, time):
    try:
        val = crash_ctr['confirmed'] / total_num_muts
    except ZeroDivisionError:
        val = 0

    data['total'].append({
        'fuzzer': fuzzer,
        'prog': prog,
        'time': time,
        'confirmed': crash_ctr['confirmed'],
        'covered': crash_ctr['covered'],
        'total': total_num_muts,
        'percentage': val * 100,
    })

    try:
        val = crash_ctr['confirmed'] / crash_ctr['covered']
    except ZeroDivisionError:
        val = 0

    data['covered'].append({
        'fuzzer': fuzzer,
        'prog': prog,
        'time': time,
        'confirmed': crash_ctr['confirmed'],
        'covered': crash_ctr['covered'],
        'total': total_num_muts,
        'percentage': val * 100,
    })

def plot(title, mut_type, data):
    import inspect
    import types
    from typing import cast
    import altair as alt
    func_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    selection = alt.selection_multi(fields=['fuzzer', 'prog'], bind='legend')
    color = alt.condition(selection,
                      alt.Color('fuzzer:O', legend=None),
                      alt.value('lightgray'))
    plot = alt.Chart(data).mark_line(
        interpolate='step-after',
    ).encode(
        x='time',
        y='percentage',
        color='fuzzer',
        tooltip=['time', 'confirmed', 'percentage', 'covered', 'total', 'fuzzer', 'prog'],
    ).properties(
        title=title,
        width=600,
        height=400,
    )

    plot = plot.mark_point(size=100, opacity=.5, tooltip=alt.TooltipContent("encoding")) + plot

    plot = plot.add_selection(
        alt.selection_interval(bind='scales')
    ).transform_filter(
        selection
    )
    
    plot = (plot |
    alt.Chart(data).mark_point().encode(
        x=alt.X('prog', axis=alt.Axis(orient='bottom')),
        y=alt.Y('fuzzer', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(
        selection
    ))

    res = f'<div id="{title.replace(" ", "")}{func_name}{mut_type}"></div>'
    res += '''<script type="text/javascript">
                vegaEmbed('#{title}{func_name}{mut_id}', {spec1}).catch(console.error);
              </script>'''.format(title=title.replace(" ", ""), func_name=func_name, mut_id=mut_type,
                                  spec1=plot.to_json(indent=None))
    return res

def split_vals(val):
    if val is None:
        return []
    return [float(v) for v in val.split("/////")]

def gather_plot_data(runs, run_results):
    from collections import defaultdict
    import pandas as pd
    if len(run_results) == 0:
        return None

    data = defaultdict(list)
    unique_crashes = [] 

    for crash in run_results.itertuples():
        import math
        if math.isnan(crash.covered_file_seen):
            continue
        unique_crashes.append({
            'fuzzer': crash.fuzzer,
            'prog': crash.prog,
            'mut_type': crash.mut_type,
            'type': 'covered',
            'time': crash.covered_file_seen,
        })

    for crash in run_results.itertuples():
        if crash.confirmed != 1:
            continue
        if crash.stage == 'initial' or crash.stage == 'runtime' or crash.stage == 'final':
            unique_crashes.append({
                'fuzzer': crash.fuzzer,
                'prog': crash.prog,
                'mut_type': crash.mut_type,
                'type': 'afl',
                'time': crash.time_found,
            })
        else:
            print(crash.stage)
            # nothing found
            pass

    counters = defaultdict(lambda: {
        'covered': 0,
        'confirmed': 0,
    })
    max_time = 0

    # for crash in unique_crashes:
    #     crash_ctr = counters[(
    #         crash['fuzzer'],
    #         crash['prog'],
    #         crash['mut_type'])]

    #     if crash['type'] == 'initial':
    #         crash_ctr['confirmed'] += 1

    # add initial points
    for run in runs.itertuples():
        crash_ctr = counters[(
            run.fuzzer,
            run.prog,
            run.mut_type)]
        total_runs = run.done
        add_datapoint(crash_ctr, data, run.mut_type, run.fuzzer, run.prog, total_runs, 0)

    for crash in sorted(unique_crashes, key=lambda x: x['time']):
        crash_ctr = counters[(
            crash['fuzzer'],
            crash['prog'],
            crash['mut_type'])]

        if crash['time'] > max_time:
            max_time = crash['time']

        total_runs = int(runs[(runs.fuzzer == crash['fuzzer']) & (runs.prog == crash['prog']) & (runs.mut_type == crash['mut_type'])]['done'].values[0])
        if crash['type'] == 'covered':
            crash_ctr['covered'] += 1
            add_datapoint(crash_ctr, data, crash['mut_type'],
                      crash['fuzzer'], crash['prog'],
                      total_runs, crash['time'])
        elif crash['type'] == 'initial':
            crash_ctr['confirmed'] += 1
            add_datapoint(crash_ctr, data, crash['mut_type'],
                      crash['fuzzer'], crash['prog'],
                      total_runs, crash['time'])
        elif crash['type'] == 'afl':
            crash_ctr['confirmed'] += 1
            add_datapoint(crash_ctr, data, crash['mut_type'],
                      crash['fuzzer'], crash['prog'],
                      total_runs, crash['time'])
        else:
            raise ValueError("Unknown type")

    # add final points
    for (fuzzer, prog, mut_type), crash_ctr in counters.items():
        try:
            total_runs = int(runs[(runs.fuzzer == fuzzer) & (runs.prog == prog) & (runs.mut_type == mut_type)]['done'].values[0])
        except ValueError:
            total_runs = 0
        add_datapoint(crash_ctr, data, mut_type, fuzzer, prog, total_runs, max_time)

    return {'total': pd.DataFrame(data['total']), 'covered': pd.DataFrame(data['covered'])}

def create_mut_type_plot(mut_type, runs, run_results):
    plot_data = gather_plot_data(runs, run_results)

    res = f'<h3>Mutation: {mut_type}</h3>'
    res += runs.to_html()
    if plot_data is not None:
        res += plot(f"Covered {mut_type}", mut_type, plot_data['covered'])
        res += plot(f"Total {mut_type}", mut_type, plot_data['total'])
    return res

def footer():
    return """
    </body>
    </html>
    """

def generate_plots():
    import pandas as pd

    con = sqlite3.connect("/home/philipp/stats.db")
    con.isolation_level = None
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    with open("eval.sql", "rt") as f:
        cur.executescript(f.read())

    res = header()
    res += runtime_stats(con)
    res += mut_stats(con)

    mut_types = pd.read_sql_query(
        "SELECT * from mut_types", con)
    runs = pd.read_sql_query(
        "select * from run_results_by_mut_type", con)
    run_results = pd.read_sql_query(
        "select * from run_results", con)
    res += "<h2>Plots</h2>"
    for mut_type in mut_types['mut_type']:
        print(mut_type)
        res += create_mut_type_plot(mut_type, 
            runs[runs.mut_type == mut_type],
            run_results[run_results.mut_type == mut_type],
        )
    res += footer()

    out_path = Path("test_plot.html").resolve()
    print(f"Writing plots to: {out_path}")
    with open(out_path, 'w') as f:
        f.write(res) 
    print(f"Open: file://{out_path}")

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
        help="run the full evaluation executing"
             "fuzzers and gathering the resulting data")
    parser.add_argument("--plots", action="store_true",
        help="generate plots for the gathered data")

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.eval:
        run_eval()
    if args.plots:
        generate_plots()

if __name__ == "__main__":
    main()
