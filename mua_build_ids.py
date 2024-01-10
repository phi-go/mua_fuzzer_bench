#!/usr/bin/env python3

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import os, argparse, random, subprocess, json
import shlex
import sqlite3
from pathlib import Path
import tempfile
import threading
import time
import traceback
import uuid
import multiprocessing


MAPPED_DIR = Path('/mapped/')
LOCATOR_TIMEOUT = 30
MAX_BUILD_TIME = 5*60 # 5 minutes
LOCATOR_MEM_LIMIT = 512 * 1024  # 512 MB
BUILD_MEM_LIMIT = 2 * 1024 * 1024  # 2 GB

BUILD_RES_OK = "compiled"
BUILD_RES_ERR = "compile_error"

@dataclass
class CompilationStatus:
    exec_id: int
    done: bool


class CompileDB:
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
    def transaction(self, transaction_type):
        while self.keep_running.is_set():
            try:
                with self.cur() as cur:
                    cur.execute(f'BEGIN {transaction_type} TRANSACTION')
                    yield cur
                    return
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    time.sleep(random.random() * 10)
                else:
                    raise
        raise Exception("Could not begin transaction.")

    def initialize(self):
        print(f"\tInitializing compile db {self.db_file}")
        with self.cur() as cur:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS compilations (
                    exec_id BLOB,
                    fuzz_target TEXT,
                    mut_id INTEGER,
                    done INTEGER,
                    build_error TEXT,
                    build_error_msg TEXT,
                    PRIMARY KEY (exec_id, fuzz_target, mut_id)
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS compiling (
                    exec_id BLOB,
                    fuzz_target TEXT,
                    mut_id INTEGER,
                    PRIMARY KEY (exec_id, fuzz_target, mut_id)
                )
            ''')
                        
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_compilations_fuzz_target ON compilations
                        (fuzz_target, mut_id, exec_id, done)
            ''')
            
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_exec_id_done ON compilations
                        (exec_id, done)
            ''')

    def get_compilation_status(self, fuzz_target, mut_id):
        with self.cur() as cur:
            cur.execute(
                'SELECT exec_id, done FROM compilations WHERE fuzz_target=? AND mut_id=?', (fuzz_target, mut_id))
            res = cur.fetchall()
        all = []
        for r in res:
            all.append(CompilationStatus(r[0], r[1]))
        return all

    def is_compiled(self, fuzz_target, mut_id):
        with self.cur() as cur:
            cur.execute(
                'SELECT * FROM compilations WHERE fuzz_target=? AND mut_id=? AND done=1',
                (fuzz_target, mut_id))
            res = cur.fetchall()
        return res is not None
        
    def get_num_compiled(self, fuzz_target):
        with self.cur() as cur:
            cur.execute(
                'SELECT COUNT(*) FROM compilations WHERE done=1 AND fuzz_target=?',
                (fuzz_target,))
            return cur.fetchone()[0]

    def try_set_started_building(self, fuzz_target, mut_id, exec_id, max_num_compiling, debug_num_mutants):
        def is_compiled(cur, fuzz_target, mut_id):
            cur.execute(
                'SELECT exec_id, done FROM compilations WHERE fuzz_target=? AND mut_id=?', (fuzz_target, mut_id))
            res = cur.fetchall()
            all = []
            for r in res:
                all.append(CompilationStatus(r[0], r[1]))
            for cs in all:
                if cs.done is True:
                    # mutant is already compiled
                    return True
                if cs.exec_id == exec_id:
                    # already compiling
                    return True
            return False

        with self.transaction('EXCLUSIVE') as cur:
            if debug_num_mutants is not None:
                cur.execute(
                    'SELECT COUNT(*) FROM compilations WHERE done=1 AND fuzz_target=? or exec_id=? AND fuzz_target=?',
                    (fuzz_target, exec_id, fuzz_target))
                num_compiled = cur.fetchone()[0]
                if num_compiled >= debug_num_mutants:
                    cur.execute('ROLLBACK TRANSACTION')
                    return "debug_num_mutants"

            cur.execute(
                'SELECT COUNT(*) FROM compilations WHERE exec_id=? AND done=0',
                (exec_id,))
            cur_compiling = cur.fetchone()[0]

            if cur_compiling < max_num_compiling:

                if is_compiled(cur, fuzz_target, mut_id):
                    cur.execute('ROLLBACK TRANSACTION')
                    return "started"

                cur.execute('INSERT INTO compilations (fuzz_target, mut_id, exec_id, done) VALUES (?, ?, ?, 0)',
                            (fuzz_target, mut_id, exec_id))
                cur.execute('COMMIT TRANSACTION')
                return "go"
            else:
                cur.execute('ROLLBACK TRANSACTION')
                return "later"

    def set_compiling(self, fuzz_target, mut_id, exec_id):
        with self.transaction('IMMEDIATE') as cur:
            cur.execute('INSERT INTO compiling (fuzz_target, mut_id, exec_id) VALUES (?, ?, ?)',
                        (fuzz_target, mut_id, exec_id))
            cur.execute('COMMIT TRANSACTION')

    def set_finished_building(self, fuzz_target, mut_id, exec_id, build_error, build_error_msg):
        with self.transaction('IMMEDIATE') as cur:
            cur.execute('UPDATE compilations SET done=1, build_error=?, build_error_msg=? WHERE fuzz_target=? AND mut_id=? AND exec_id=?',
                        (build_error, build_error_msg, fuzz_target, mut_id, exec_id))
            cur.execute('COMMIT TRANSACTION')


def estimate_max_num_compiling():
    with open('/proc/meminfo') as file:
        for line in file:
            if 'MemTotal' in line:
                mem_in_kb = line.split()[1]
                break

    usable_mem = int(mem_in_kb)
    max_compilation_mem = 2 * 1000 * 1000 # ~2 GB
    max_mem_concurrent = int(int(usable_mem) / max_compilation_mem)

    # Get the number of cores
    cpu_count = multiprocessing.cpu_count()
    max_cpu_concurrent = cpu_count
    print(f"\tusable_mem: {usable_mem} kB, cpu_count: {cpu_count}")

    return min(int(max_mem_concurrent), int(max_cpu_concurrent))


def build_mutant(mut_id, exec_id, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants, max_num_compiling,
                 keep_running):
    start_time = time.time()
    compile_db = CompileDB(compile_db_path, keep_running)

    mutant_file = mutants_dir/f'{mut_id}'

    # mark mutant as being compiled
    while True:
        res = compile_db.try_set_started_building(fuzz_target, mut_id, exec_id, max_num_compiling, debug_num_mutants)
        if res == "go":
            break
        elif res == "started":
            return ("started", time.time() - start_time)
        elif res == "later":
            time.sleep(random.random() * 10)
        elif res == "debug_num_mutants":
            return ("debug_num_mutants reached", time.time() - start_time)
        else:
            raise Exception(f"Unknown res: {res}")
    compile_start_time = time.time()

    compile_db.set_compiling(fuzz_target, mut_id, exec_id)

    config_file = mutant_file.with_suffix(".json")
    
    # create config file
    with open(config_file, "w") as f:
        json.dump([{
            "prog": fuzz_target,
            "mutation_ids": [mut_id],
            "mode": "single",
        }], f)
    
    # build new mutant and store it in mutants_dir
    build_cmd = [
        'pipx',
        'run',
        'hatch',
        'run',
        'src/mua_fuzzer_benchmark/eval.py',
        'locator_mutants_local',
        '--result-path',
        mutant_file,
        '--statsdb',
        '/mua_build/build/stats.db',
        '--mutation-list',
        config_file
    ]
    full_cmd = f'ulimit -v {BUILD_MEM_LIMIT} ; {shlex.join((str(cc) for cc in build_cmd))}'
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env['MUT_SHARED_DIR'] = tmp_dir
        env['MUT_HOST_TMP_PATH'] = tmp_dir
        try:
            subprocess.run(
                full_cmd,
                cwd='/mutator',
                env=env,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                errors='replace',
                check=True,
                timeout=MAX_BUILD_TIME,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            # set as finished even for errors
            compile_db.set_finished_building(fuzz_target, mut_id, exec_id, "process_error", e.output)
            return (BUILD_RES_ERR, time.time() - start_time, time.time() - compile_start_time, e.output)
        except subprocess.TimeoutExpired as e:
            # set as finished even for errors
            compile_db.set_finished_building(fuzz_target, mut_id, exec_id, "timeout", e.output)
            return (BUILD_RES_ERR, time.time() - start_time, time.time() - compile_start_time, e.output)
        except Exception as e:
            # set as finished even for errors
            compile_db.set_finished_building(fuzz_target, mut_id, exec_id, "exception", str(e))
            return (BUILD_RES_ERR, time.time() - start_time, time.time() - compile_start_time, str(e))
        finally:
            #cleanup
            os.remove(config_file)

    # set as finished even for errors
    compile_db.set_finished_building(fuzz_target, mut_id, exec_id, None, None)
    # mark mutant as compiled
    return (BUILD_RES_OK, time.time() - start_time, time.time() - compile_start_time)


def locate_corpus_entry(corpus_entry, locator_path, corpus_dir, mutants_ids_dir):
    input_file = os.path.join(corpus_dir, corpus_entry)
    mutant_ids_result_file = (mutants_ids_dir / f"{Path(corpus_entry).name}.json")
    if os.path.isdir(input_file):
        return
    if mutant_ids_result_file.is_file():
        return
    mutant_ids_result_file.touch()
    corpus_mut_ids = []
    all_mut_ids = set()
    timed_out = None
    errored_out = None
    cmd = [str(locator_path), str(input_file)]
    full_cmd = f'ulimit -v {LOCATOR_MEM_LIMIT} ; {shlex.join(cc for cc in cmd)}'
    with tempfile.TemporaryDirectory() as trigger_dir:
        try:
            subprocess.run(
                full_cmd,
                cwd=trigger_dir,
                timeout=LOCATOR_TIMEOUT,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                errors='replace',
                check=True,
                shell=True,
            )
        except subprocess.TimeoutExpired as e:
            timed_out = str(e)
        except Exception as e:
            errored_out = str(e)
        for mut_id_file in (Path(trigger_dir)/'trigger_signal').glob("*"):
            # if not mut_id_file.is_file():
            #     print(f"Mutant id file is not a file: {mut_id_file}")
            #     continue
            file_name = mut_id_file.stem
            try:
                mut_id = int(file_name)
            except ValueError:
                print(f"\tInvalid mutant id: {file_name}")
                continue
            corpus_mut_ids.append(mut_id)
            all_mut_ids.add(mut_id)
    with open(mutant_ids_result_file, 'w') as f:
        json.dump({
            'mut_ids': corpus_mut_ids,
            'errored_out': errored_out,
            'timed_out': timed_out,
        }, f)
    return all_mut_ids, timed_out, errored_out


def keep_running_initializer(keep_running):
    global KEEP_RUNNING
    KEEP_RUNNING = keep_running


def main():
    parser = argparse.ArgumentParser(
                    prog='mua_build_ids',
                    description='Script runs generated corpus against locator targets to get a list of covered mutants. Then, covered mutants are build, if not already existent',
                    epilog='')

    parser.add_argument('exec_id', metavar='I',
                    help='fuzzbench fuzz_target')

    parser.add_argument('fuzz_target', metavar='T',
                    help='fuzzbench fuzz_target')

    parser.add_argument('benchmark', metavar='B',
                    help='fuzzbench benchmark')

    parser.add_argument('experiment', metavar='E',
                    help='name of the fuzzbench experiment')

    parser.add_argument('fuzzer', metavar='F',
                    help='name of currently processed fuzzer')

    parser.add_argument('trial_num', metavar='N',
                    help='fuzzbench trial_num')

    parser.add_argument('--debug_num_mutants', metavar='D', required=False, type=int,
                    help='For debugging purposes, limit the number of mutants to build.')

    args = parser.parse_args()

    exec_id = uuid.UUID(args.exec_id)
    fuzz_target = args.fuzz_target
    benchmark = args.benchmark
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)
    debug_num_mutants = args.debug_num_mutants

    shared_mua_binaries_dir = MAPPED_DIR / experiment / 'mua-results'
    compile_db_path = shared_mua_binaries_dir / 'compile.sqlite'
    corpus_dir = shared_mua_binaries_dir / 'corpi' / fuzzer / trial_num
    mutants_ids_dir = shared_mua_binaries_dir / 'mutant_ids' / benchmark / fuzzer / trial_num
    mutants_ids_dir.mkdir(parents=True, exist_ok=True)

    mutants_dir = shared_mua_binaries_dir / 'mutants' / benchmark
    mutants_dir.mkdir(parents=True, exist_ok=True)
    locator_path = (Path('/out') / fuzz_target).with_suffix('.locator')

    max_num_compiling = estimate_max_num_compiling()
    print(f"\tEstimating maximum number of concurrent compilations: {max_num_compiling}")

    keep_running = threading.Event()
    keep_running.set()
    compile_db = CompileDB(compile_db_path, keep_running)
    compile_db.initialize()

    all_mut_ids = set()

    locate_start_time = time.time()

    # execute corpus with locator
    corpus_list = os.listdir(corpus_dir)
    corpus_len = len(corpus_list)
    locate_jobs = {}
    completed_count = 0
    successful_count = 0
    timeout_count = 0
    errored_count = 0
    max_jobs = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        while True:
            while len(locate_jobs) < max_jobs and len(corpus_list) > 0:
                corpus_entry = corpus_list.pop()
                locate_jobs[executor.submit(
                    locate_corpus_entry, corpus_entry, locator_path, corpus_dir, mutants_ids_dir
                )] = "locate"

            if len(corpus_list) == 0 and len(locate_jobs) == 0:
                break

            job = next(as_completed(locate_jobs))
            job_meta = locate_jobs.pop(job)
            job_type = job_meta
            if job_type == "locate":
                try:
                    res = job.result()
                except Exception as e:
                    stacktrace = traceback.format_exc()
                    print(f"\tError while locating corpus entry: {e}:\n{stacktrace}")
                    res = None
                if res is not None:
                    found_mut_ids, timed_out, errored_out = res
                    all_mut_ids.update(found_mut_ids)
                    if timed_out is not None:
                        timeout_count += 1
                    if errored_out is not None:
                        errored_count += 1
                    if not (timed_out or errored_out):
                        successful_count += 1
                completed_count += 1
            else:
                raise Exception(f"Unknown job type: {job_type}")

    print(f"\tLocator: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}:")
    print(f"\t{completed_count}/{corpus_len} corpus entries in {time.time() - locate_start_time:.2f}s.")
    print(f"\tNew results: {successful_count} successful, {timeout_count} timed out, {errored_count} errored out.")
    print(f"\tNum of mutants covered: {len(all_mut_ids)}")

    exec_id_bytes = exec_id.bytes

    if debug_num_mutants is not None and len(all_mut_ids) > debug_num_mutants:
        print(f"\tfor debugging reducing num of all_mut_ids: {len(all_mut_ids)} to {debug_num_mutants}")
        all_mut_ids = set(random.sample(list(all_mut_ids), debug_num_mutants))
        print(f"\tall_mut_ids: {len(all_mut_ids)}")

    build_start_time = time.time()

    mutations_to_compile = set()
    # build mutants
    job_results = Counter()
    total_build_time = 0
    total_thread_time = 0
    build_jobs = {}
    max_num_compile = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=max_num_compile) as executor:
        while True:
            while len(build_jobs) < max_num_compile and len(all_mut_ids) > 0:
                mut_id = all_mut_ids.pop()
                mutations_to_compile.add(mut_id)
                build_jobs[executor.submit(
                    build_mutant, mut_id, exec_id_bytes, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants, max_num_compiling, keep_running
                )] = ("build", mut_id)

            if len(build_jobs) == 0 and len(all_mut_ids) == 0:
                break

            job = next(as_completed(build_jobs))
            job_meta = build_jobs.pop(job)
            job_type = job_meta[0]
            if job_type == "build":
                try:
                    res = job.result()
                except Exception as e:
                    stacktrace = traceback.format_exc()
                    print(f"\tError while building mutant: {e}:\n{stacktrace}")
                    res = None
                if res is not None:
                    res_type = res[0]
                    job_results[res_type] += 1
                    if res_type in [BUILD_RES_OK, BUILD_RES_ERR]:
                        total_build_time += res[2]
                        mut_id = job_meta[1]
                        mutations_to_compile.remove(mut_id)
                    if res_type in ['started', 'debug_num_mutants reached', BUILD_RES_OK, BUILD_RES_ERR]:
                        total_thread_time += res[1]
            else:
                raise Exception(f"Unknown job type: {job_type}")


    if debug_num_mutants is not None:
        while True:
            num_compiled = compile_db.get_num_compiled(fuzz_target)
            if num_compiled >= debug_num_mutants or num_compiled >= len(all_mut_ids):
                break
            time.sleep(1)
    else:
        start_wait_for_compilations = time.time()
        # wait at most build time, all build jobs should have been started if we get here
        while start_wait_for_compilations - time.time() < MAX_BUILD_TIME: 
            finished_compiling = set()
            for mut_id in mutations_to_compile:
                if compile_db.is_compiled(fuzz_target, mut_id):
                    finished_compiling.add(mut_id)
            mutations_to_compile -= finished_compiling
            if len(mutations_to_compile) == 0:
                break
            time.sleep(1)
        if len(mutations_to_compile) > 0:
            print(
                "Warning: timeout while waiting for all mutants to be compiled: " +
                f"{len(mutations_to_compile)} mutants left")

    print(f"\tBuilder: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}: {len(all_mut_ids)} mutants in {time.time() - build_start_time:.2f}s.")
    print("\tBuilder job results:")
    for k, v in job_results.most_common():
        print(f"\t\t{k}: {v}")
    print(f"\ttotal build time: {total_build_time:.2f}s (single core)")
    print(f"\ttotal thread time: {total_thread_time:.2f}s (thread)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\tError: {e}")
        traceback.print_exc()
        raise e
