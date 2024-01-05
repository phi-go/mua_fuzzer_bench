#!/usr/bin/env python3

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os, argparse, random, subprocess, json
import sqlite3
from pathlib import Path
from multiprocessing import Pool
import tempfile
import time
import traceback
import uuid
import multiprocessing


MAPPED_DIR = Path('/mapped/')
LOCATOR_TIMEOUT = 30
MAX_BUILD_TIME = 5*60 # 5 minutes


@dataclass
class CompilationStatus:
    exec_id: int
    done: bool


class CompileDB:
    def __init__(self, db_file):
        self.db_file = db_file

    @contextmanager
    def connect(self):
        with sqlite3.connect(self.db_file, check_same_thread=False, timeout=300) as conn:
            yield conn

    def initialize(self):
        print(f"\tInitializing compile db {self.db_file}")
        with self.connect() as conn:
            cur = conn.cursor()
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
            conn.commit()

    def get_compilation_status(self, fuzz_target, mut_id):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                'SELECT exec_id, done FROM compilations WHERE fuzz_target=? AND mut_id=?', (fuzz_target, mut_id))
            res = cur.fetchall()
        all = []
        for r in res:
            all.append(CompilationStatus(r[0], r[1]))
        return all

    def is_compiled(self, fuzz_target, mut_id):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                'SELECT * FROM compilations WHERE fuzz_target=? AND mut_id=? AND done=1',
                (fuzz_target, mut_id))
            res = cur.fetchall()
        return res is not None
        
    def get_num_compiled(self, fuzz_target):
        with self.connect() as conn:
            cur = conn.cursor()
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

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute('BEGIN EXCLUSIVE TRANSACTION')

            if debug_num_mutants is not None:
                cur = conn.cursor()
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
                conn.execute('COMMIT TRANSACTION')
                return "go"
            else:
                cur.execute('ROLLBACK TRANSACTION')
                return "later"

    def set_compiling(self, fuzz_target, mut_id, exec_id):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO compiling (fuzz_target, mut_id, exec_id) VALUES (?, ?, ?)',
                        (fuzz_target, mut_id, exec_id))
            conn.commit()

    def set_finished_building(self, fuzz_target, mut_id, exec_id, build_error, build_error_msg):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute('UPDATE compilations SET done=1, build_error=?, build_error_msg=? WHERE fuzz_target=? AND mut_id=? AND exec_id=?',
                        (build_error, build_error_msg, fuzz_target, mut_id, exec_id))
            conn.commit()


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


def build_mutant(mut_id, exec_id, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants, max_num_compiling):
    start_time = time.time()
    compile_db = CompileDB(compile_db_path)

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
    build_command = [
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env['MUT_SHARED_DIR'] = tmp_dir
        env['MUT_HOST_TMP_PATH'] = tmp_dir
        try:
            subprocess.run(
                build_command,
                cwd='/mutator',
                env=env,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                errors='replace',
                check=True,
                timeout=MAX_BUILD_TIME,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while building mutant: {e.output}")
            # set as finished even for errors
            compile_db.set_finished_building(fuzz_target, mut_id, exec_id, "error", e.output)
            return ("compile_error", time.time() - start_time, time.time() - compile_start_time, e.output)
        except subprocess.TimeoutExpired as e:
            print(f"Error while building mutant: {e.output}")
            # set as finished even for errors
            compile_db.set_finished_building(fuzz_target, mut_id, exec_id, "timeout", e.output)
            return ("compile_error", time.time() - start_time, time.time() - compile_start_time, e.output)
        finally:
            #cleanup
            os.remove(config_file)

    # set as finished even for errors
    compile_db.set_finished_building(fuzz_target, mut_id, exec_id, None, None)
    # mark mutant as compiled
    return ("compiled", time.time() - start_time, time.time() - compile_start_time)


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
    with tempfile.TemporaryDirectory() as trigger_dir:
        try:
            subprocess.run(
                [str(locator_path), str(input_file)],
                cwd=trigger_dir,
                timeout=LOCATOR_TIMEOUT,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                errors='replace',
                check=True,
            )
        except subprocess.CalledProcessError as e:
            errored_out = str(e)
        except subprocess.TimeoutExpired as e:
            timed_out = str(e)
        for mut_id_file in (Path(trigger_dir)/'trigger_signal').glob("*"):
            if not mut_id_file.is_file():
                print(f"Mutant id file is not a file: {mut_id_file}")
                continue
            file_name = mut_id_file.stem
            try:
                mut_id = int(file_name)
            except ValueError:
                print(f"Invalid mutant id: {file_name}")
                return
            corpus_mut_ids.append(mut_id)
            all_mut_ids.add(mut_id)
    with open(mutant_ids_result_file, 'w') as f:
        json.dump({
            'mut_ids': corpus_mut_ids,
            'errored_out': errored_out,
            'timed_out': timed_out,
        }, f)
    return all_mut_ids, timed_out, errored_out


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

    compile_db = CompileDB(compile_db_path)
    compile_db.initialize()

    all_mut_ids = set()

    locate_start_time = time.time()

    # execute corpus with locator
    corpus_list = os.listdir(corpus_dir)
    corpus_len = len(corpus_list)
    locate_jobs = []
    with ThreadPoolExecutor(max_workers=int(multiprocessing.cpu_count())) as executor:
        for corpus_entry in corpus_list:
            locate_jobs.append(executor.submit(
                locate_corpus_entry, corpus_entry, locator_path, corpus_dir, mutants_ids_dir))

    completed_count = 0
    successful_count = 0
    timeout_count = 0
    errored_count = 0
    for job in as_completed(locate_jobs):
        try:
            res = job.result()
        except Exception as e:
            stacktrace = traceback.format_exc()
            print(f"Error while locating corpus entry: {e}:\n{stacktrace}")
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

    print(f"\tLocator: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}:")
    print(f"\t{completed_count}/{corpus_len} corpus entries in {time.time() - locate_start_time:.2f}s.")
    print(f"\tNew results: {successful_count} successful, {timeout_count} timed out, {errored_count} errored out.")
    print(f"\tNum of mutants covered: {len(all_mut_ids)}")

    exec_id_bytes = exec_id.bytes

    if debug_num_mutants is not None and len(all_mut_ids) > debug_num_mutants:
        print(f"\tfor debugging reducing num of all_mut_ids: {len(all_mut_ids)} to {debug_num_mutants}")
        all_mut_ids = random.sample(list(all_mut_ids), debug_num_mutants)
        print(f"\tall_mut_ids: {len(all_mut_ids)}")

    build_start_time = time.time()

    mutations_to_compile = set()
    # build mutants
    build_jobs = []
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
        for mut_id in all_mut_ids:
            mutations_to_compile.add(mut_id)
            build_jobs.append(executor.submit(build_mutant, mut_id, exec_id_bytes, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants, max_num_compiling))

    job_results = Counter()
    total_build_time = 0
    total_thread_time = 0
    for job in as_completed(build_jobs):
        try:
            res = job.result()
        except Exception as e:
            stacktrace = traceback.format_exc()
            print(f"Error while building mutant: {e}:\n{stacktrace}")

        res_type = res[0]
        job_results[res_type] += 1
        if res_type in ['compiled', 'compile_error']:
            total_build_time += res[2]
        if res_type in ['started', 'debug_num_mutants reached', 'compile_error', 'compiled']:
            total_thread_time += res[1]

    if debug_num_mutants is not None:
        while True:
            num_compiled = compile_db.get_num_compiled(fuzz_target)
            if num_compiled >= debug_num_mutants:
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
        print(f"Error: {e}")
        traceback.print_exc()
        raise e
