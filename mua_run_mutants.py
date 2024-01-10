#!/usr/bin/env python3

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import os, argparse, subprocess, json
import shlex
import random
from pathlib import Path

import tempfile
import threading
import multiprocessing
import time
import traceback
import sqlite3

MAPPED_DIR = Path('/mapped/')
RUN_TIMEOUT = 1
ID_CHUNK_SIZE = 1
RUN_CHUNK_SIZE = 35
MAX_RUNNER_RUNTIME = RUN_CHUNK_SIZE * RUN_TIMEOUT * 2
MEM_LIMIT = 512 * 1024  # 512 MB


@dataclass
class RunInputResult:
    killed: bool
    orig_retcode: int
    mutant_retcode: int
    original_runtime: float
    mutant_runtime: float
    original_timed_out: bool
    mutant_timed_out: bool

class ResultDB:
    def __init__(self, db_file, keep_running):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=300)
        self.keep_running = keep_running

    @contextmanager
    def cur(self):
        with self.conn as conn:
            cur = conn.cursor()
            yield cur
            cur.close()

    @contextmanager
    def immediate_transaction(self):
        while self.keep_running.is_set():
            try:
                with self.cur() as cur:
                    cur.execute('BEGIN IMMEDIATE TRANSACTION')
                    yield cur
                    return
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    time.sleep(random.random() * 10)
                else:
                    raise
        raise Exception("Could not begin transaction.")

    def initialize(self, benchmark, fuzz_target, fuzzer, trial_num):
        with self.cur() as cur:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    input_file_id INTEGER,
                    mut_id INTEGER,
                    skipped INTEGER,
                    killed INTEGER,
                    orig_retcode INTEGER,
                    mutant_retcode INTEGER,
                    orig_runtime REAL,
                    mutant_runtime REAL,
                    orig_timed_out INTEGER,
                    mutant_timed_out INTEGER,
                    PRIMARY KEY (input_file_id, mut_id)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS run_info (
                    benchmark TEXT,
                    fuzz_target TEXT,
                    fuzzer TEXT,
                    trial_num TEXT,
                    UNIQUE(benchmark, fuzz_target, fuzzer, trial_num)
                )
            ''')
            cur.execute('''
                INSERT OR IGNORE INTO run_info(benchmark, fuzz_target, fuzzer, trial_num)
                        VALUES (?, ?, ?, ?)
            ''', (benchmark, fuzz_target, fuzzer, trial_num))
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_results ON results (input_file_id, mut_id, killed)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_mut_is_kill ON results (mut_id, killed)
            ''')
            cur.execute('''COMMIT''')

    def get_all_done_for_file(self, input_file_id):
        with self.cur() as cur:
            cur.execute('''
                SELECT mut_id FROM results WHERE input_file_id = ?
            ''', (input_file_id,))
            res = cur.fetchall()
        return set(r[0] for r in res)

    def get_earliest_timestamp_killing_mutant(self, mut_id):
        with self.cur() as cur:
            cur.execute('''
                SELECT timestamp FROM timestamps
                JOIN results USING (input_file_id)
                WHERE mut_id = ? AND killed = 1
                ORDER BY timestamp
                LIMIT 1
            ''', (mut_id,))
            res = cur.fetchone()
        if res is None:
            return None
        else:
            return res[0]

    def get_info_for_hashname(self, hashname):
        with self.cur() as cur:
            cur.execute('''
                SELECT input_file_id, timestamp FROM timestamps WHERE hashname = ?
                LIMIT 1
            ''', (hashname,))
            res = cur.fetchone()
        if res is None:
            return None
        else:
            return res

    def add_all_results(self, results):
        with self.immediate_transaction() as cur:
            for input_file_id, mut_id, res in results:
                cur.execute('''
                    INSERT OR IGNORE INTO results (
                            input_file_id,
                            mut_id,
                            skipped,
                            killed,
                            orig_retcode,
                            mutant_retcode,
                            orig_runtime,
                            mutant_runtime,
                            orig_timed_out,
                            mutant_timed_out
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    input_file_id,
                    mut_id,
                    False,
                    res.killed,
                    res.orig_retcode,
                    res.mutant_retcode,
                    res.original_runtime,
                    res.mutant_runtime,
                    res.original_timed_out,
                    res.mutant_timed_out,
                ))
            cur.execute('COMMIT TRANSACTION')

    def add_all_skipped(self, skipped):
        with self.immediate_transaction() as cur:
            for input_file_id, mut_id in skipped:
                cur.execute('''
                    INSERT OR IGNORE INTO results (input_file_id, mut_id, skipped)
                        VALUES (?, ?, ?)
                ''', (input_file_id, mut_id, True))
            cur.execute('COMMIT TRANSACTION')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run(  # type: ignore[misc]
        cmd,
        timeout,
        **kwargs,
):
    """
    Run command locally.
    If return_code is not 0, raise a ValueError containing the run result.
    """
    start_time = time.time()
    timed_out = False
    full_cmd = f'ulimit -v {MEM_LIMIT} ; exec {shlex.join((str(cc) for cc in cmd))}'
    with subprocess.Popen(full_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # close_fds=True,
        # errors='backslashreplace',  # text mode: stdout is a str
        # preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        shell=True,
        **kwargs  # type: ignore[misc]
    ) as proc:
        try:
            try:
                _ = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                timed_out = True
                _ = proc.wait(timeout=1)
            except PermissionError:
                proc.kill()
                _ = proc.wait(timeout=1)
            except:
                proc.kill()
                _ = proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass

        returncode = proc.poll()
        runtime = time.time() - start_time

        return {
            'returncode': returncode,
            # 'out': stdout.decode('utf-8', 'backslashreplace'),
            'timed_out': timed_out,
            'runtime': runtime,
        }


def run_input(tmpdir, input_file, original_executable, mutant_executable):
    # execute input_file on original executable
    original_result = run([original_executable, input_file], timeout=RUN_TIMEOUT, cwd=tmpdir)
    # execute input_file on mutant executable
    mutant_result = run([mutant_executable, input_file], timeout=RUN_TIMEOUT, cwd=tmpdir)
    # compare results
    # if results differ, add mutant to killed_mutants
    # else add mutant to surviving_mutants
    return RunInputResult(
        killed=original_result['returncode'] != mutant_result['returncode'],
        orig_retcode=original_result['returncode'],
        mutant_retcode=mutant_result['returncode'],
        original_runtime=original_result['runtime'],
        mutant_runtime=mutant_result['runtime'],
        original_timed_out=original_result['timed_out'],
        mutant_timed_out=mutant_result['timed_out'],
    )


def run_inputs(results_db, tmpdir, original_executable, mutants_dir, fuzz_target, keep_running, chunk):
    results_db = ResultDB(results_db, keep_running)
    result_counter = Counter()
    results = []
    for input_file, input_file_id, mut_id in chunk:
        mutant_executable = mutants_dir / str(mut_id) / f'{fuzz_target}_{mut_id}'
        res = run_input(
            tmpdir, input_file, original_executable, mutant_executable)
        results.append((input_file_id, mut_id, res))
        result_counter["done"] += 1

    results_db.add_all_results(results)
    return result_counter

# for corpus entry in mutants_ids_dir run entry on each covered mutant
# check if corpus entry has already been run, if so skip
# else add corpus entry for each mutant to jobs
def gather_run_jobs_chunk(
    corpus_mutant_id_files, corpus_dir, mutants_dir, fuzz_target, results_db
):
    stats = Counter()
    todo_run_jobs = []
    skipped = []
    earliest_kill_cache = {}
    for corpus_mutant_ids_file in corpus_mutant_id_files:
        if corpus_mutant_ids_file.is_dir():
            continue
        input_file = os.path.join(corpus_dir, corpus_mutant_ids_file.name[:-5])

        if not Path(input_file).is_file():
            print(f"\tcorpus mutant ids file {corpus_mutant_ids_file} has no input file {input_file}")
            stats.update(["no_input"])
            continue

        input_file_name = Path(input_file).name

        with open(corpus_mutant_ids_file, 'r') as f:
            try:
                input_mutant_ids = json.load(f)['mut_ids']
            except Exception as e:
                print(f"\tError loading json file {corpus_mutant_ids_file}: {e}")
                stats.update(["no_json"])
                continue
        hashname_info = results_db.get_info_for_hashname(input_file_name)
        if hashname_info is None:
            stats.update(["(no_timestamp)"])
            continue
            # print(f"Could not find timestamp for input file: {input_file}")
        input_file_id, file_timestamp = hashname_info
        all_done_for_file = results_db.get_all_done_for_file(input_file_id)
        for mut_id in input_mutant_ids:
            if mut_id in all_done_for_file:
                # print(f"corpus input file {input_file} has already been run")
                stats.update(["outside_already_done"])
                continue
            if mut_id in earliest_kill_cache:
                earliest_kill = earliest_kill_cache[mut_id]
            else:
                earliest_kill = results_db.get_earliest_timestamp_killing_mutant(mut_id)
                earliest_kill_cache[mut_id] = earliest_kill
            if earliest_kill is not None:
                    if file_timestamp >= earliest_kill:
                        skipped.append((input_file_id, mut_id))
                        stats["skipped"] += 1
                        continue
            mutant_executable = mutants_dir / str(mut_id) / f'{fuzz_target}_{mut_id}'
            if not mutant_executable.is_file():
                # print(f"mutant file {mutant_executable} does not exist")  # TODO this is noisy for debug
                stats.update(["no_mutant_executable"])
                continue
            todo_run_jobs.append((input_file, input_file_id, mut_id))
    return todo_run_jobs, stats, skipped


def do_run(
    corpus_dir, mutants_dir, fuzz_target, result_db_path, tmpdir, original_executable, mut_chunk
):
    keep_running = KEEP_RUNNING
    results_db = ResultDB(result_db_path, keep_running)
    todo_run_jobs, stats, skipped = gather_run_jobs_chunk(
        mut_chunk, corpus_dir, mutants_dir, fuzz_target, results_db
    )
    results_db.add_all_skipped(skipped)
    with ThreadPoolExecutor(max_workers=2) as executor:
        run_inputs_partial = partial(run_inputs,
                                     result_db_path, tmpdir, original_executable, mutants_dir, fuzz_target,
                                     keep_running)
        run_jobs = {}
        while True:
            chunk, todo_run_jobs = todo_run_jobs[:RUN_CHUNK_SIZE], todo_run_jobs[RUN_CHUNK_SIZE:]
            if len(chunk) == 0:
                break
            run_jobs[executor.submit(run_inputs_partial, chunk)] = "run"
        for job in as_completed(run_jobs):
            job_meta = run_jobs[job]
            del[run_jobs[job]]
            try:
                res = job.result()
            except Exception as e:
                stacktrace = traceback.format_exc()
                print(f"\tError during '{job_meta}' job: {e}:\n{stacktrace}")
                continue
            if job_meta == "run":
                stats.update(res)
    return stats



def split_list(lst, num_chunks):
    """Split the list |lst| into |num_chunks| chunks."""
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return chunks(lst, chunk_size)


def keep_running_initializer(keep_running):
    global KEEP_RUNNING
    KEEP_RUNNING = keep_running


def main():
    parser = argparse.ArgumentParser(
                    prog='mua_run_mutants',
                    description='Script runs corpus against all covered mutants',
                    epilog='')

    parser.add_argument('fuzz_target', metavar='T',
                    help='fuzzbench fuzz_target')

    parser.add_argument('benchmark', metavar='B',
                    help='fuzzbench fuzz_target')

    parser.add_argument('experiment', metavar='E',
                    help='name of the fuzzbench experiment')

    parser.add_argument('fuzzer', metavar='F',
                    help='name of currently processed fuzzer')

    parser.add_argument('trial_num', metavar='N',
                    help='fuzzbench trial_num')

    args = parser.parse_args()

    start_time = time.time()

    fuzz_target = args.fuzz_target
    benchmark = args.benchmark
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)
    original_executable = Path(os.environ['OUT']) / fuzz_target

    shared_mua_binaries_dir = MAPPED_DIR / experiment / 'mua-results'
    corpus_dir = shared_mua_binaries_dir / 'corpi' / fuzzer / trial_num
    mutants_ids_dir = shared_mua_binaries_dir / 'mutant_ids'/ benchmark / fuzzer / trial_num
    corpus_run_results = shared_mua_binaries_dir / 'corpus_run_results' / fuzzer / trial_num
    result_db_path = corpus_run_results / f'results.sqlite'

    keep_running_runner = threading.Event()
    keep_running_runner.set()

    keep_running_gen = multiprocessing.Event()
    keep_running_gen.set()
    results_db = ResultDB(result_db_path, keep_running_runner)
    results_db.initialize(benchmark, fuzz_target, fuzzer, trial_num)
    mutants_dir = shared_mua_binaries_dir / 'mutants' / benchmark

    print(f"\toriginal executable: {original_executable}")
    print(f"\tcorpus dir: {corpus_dir.absolute()}")

    stats = Counter()

    num_run_processes = os.cpu_count()

    mutant_id_files = list(mutants_ids_dir.glob('*'))
    print(f"\t{len(mutant_id_files)=}")
    mutant_id_files_todo = mutant_id_files.copy()
    stopping_thread = None

    last_print = time.time()
    
    # run mutants
    run_jobs_start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        do_run_partial = partial(do_run, corpus_dir, mutants_dir, fuzz_target, result_db_path, tmpdir, original_executable)
        active_run_jobs = 0
        runner_count = 0
        run_jobs = {}
        todo_list = []
        with ProcessPoolExecutor(
                 max_workers=num_run_processes, initializer=keep_running_initializer, initargs=(keep_running_gen,)
                 ) as executor:
            while True:
                while active_run_jobs <= num_run_processes:
                    mut_chunk, mutant_id_files_todo = mutant_id_files_todo[:ID_CHUNK_SIZE], mutant_id_files_todo[ID_CHUNK_SIZE:]
                    if len(mut_chunk) == 0:
                        break
                    active_run_jobs += 1
                    run_jobs[executor.submit(do_run_partial, mut_chunk)] = "gen"

                # while runner_count <= num_runner_threads \
                #         and ((len(todo_list) > RUN_CHUNK_SIZE) or (gen_count == 0 and len(todo_list) > 0)):
                #     todo_chunk, todo_list = todo_list[:RUN_CHUNK_SIZE], todo_list[RUN_CHUNK_SIZE:]
                #     run_jobs[runner_executor.submit(run_inputs_partial, todo_chunk)] = "run"
                #     runner_count += 1

                if len(mutant_id_files_todo) == 0 \
                        and len(todo_list) == 0 \
                        and active_run_jobs == 0 \
                        and stopping_thread is None:
                    # everything has been generated and all runners are started
                    # start a thread that set keep_running to false when max runtime is reached
                    def stop_running():
                        start_time = time.time()
                        while time.time() - start_time < MAX_RUNNER_RUNTIME:
                            if not keep_running_runner.is_set():
                                return
                            time.sleep(1)
                        print(f"\tClearing keep_running")
                        keep_running_gen.clear()
                        keep_running_runner.clear()

                    print(f"\tStarting stop_running thread")
                    stopping_thread = threading.Thread(target=stop_running)
                    stopping_thread.start()

                if time.time() - last_print > 1:
                    time_since_start = time.time() - run_jobs_start
                    print(f"\t{time_since_start:.2f} {active_run_jobs=}, {len(mutant_id_files_todo)=},\n\t{stats=}")
                    last_print = time.time()

                try:
                    job = next(as_completed(run_jobs))
                except StopIteration:
                    # all done
                    break
                job_meta = run_jobs[job]
                del[run_jobs[job]]
                try:
                    res = job.result()
                except Exception as e:
                    stacktrace = traceback.format_exc()
                    print(f"\tError during '{job_meta}' job: {e}:\n{stacktrace}")
                    continue
                if job_meta == "gen":
                    active_run_jobs -= 1
                    assert active_run_jobs >= 0
                    stats.update(res)
                elif job_meta == "run":
                    runner_count -= 1
                    assert runner_count >= 0
                    stats.update(res)

    keep_running_runner.clear()
    if stopping_thread is not None:
        print(f"\tWaiting for stopping_thread to finish")
        stopping_thread.join()
    print(f"\tmua run stats:")
    for msg, cnt in stats.most_common():
        print(f"\t\t{msg}: {cnt}")
    print(f"\tmua run time: {time.time() - start_time:0.2f}s")
    try:
        done_jobs = stats["done"]
        print(f"\t{done_jobs / (time.time() - run_jobs_start):0.2f} done/s")
    except KeyError:
        pass
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\tError: {e}")
        traceback.print_exc()
        raise e
