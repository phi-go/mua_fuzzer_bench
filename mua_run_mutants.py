#!/usr/bin/env python3

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os, argparse, subprocess, json
from pathlib import Path

import tempfile
import time
import traceback
import sqlite3

MAPPED_DIR = Path('/mapped/')
RUN_TIMEOUT = 1
ID_CHUNK_SIZE = 20
RUN_CHUNK_SIZE = 1000

class ResultDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=300)

    @contextmanager
    def cur(self):
        with self.conn as conn:
            cur = conn.cursor()
            yield cur
            cur.close()

    def initialize(self, benchmark, fuzz_target, fuzzer, trial_num):
        with self.cur() as cur:
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    input_file TEXT,
                    mut_id INTEGER,
                    skipped INTEGER,
                    killed INTEGER,
                    orig_res TEXT,
                    mutant_res TEXT,
                    PRIMARY KEY (input_file, mut_id)
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
                CREATE INDEX IF NOT EXISTS idx_results ON results (input_file, mut_id, killed)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_mut_is_kill ON results (mut_id, killed)
            ''')
            cur.execute('''COMMIT''')

    def input_on_mut_is_done(self, input_file, mut_id, cur=None):
        def _input_on_mut_is_done(cur):
            cur.execute('''
                SELECT 1 FROM results WHERE input_file = ? AND mut_id = ?
                LIMIT 1
            ''', (input_file, mut_id))
            res = cur.fetchone()
            return res is not None
        if cur is None:
            with self.cur() as cur:
                return _input_on_mut_is_done(cur)
        else:
            return _input_on_mut_is_done(cur)
    
    def get_all_done(self):
        with self.cur() as cur:
            cur.execute('''
                SELECT input_file, mut_id FROM results
            ''')
            res = cur.fetchall()
        return set(hash((r[0],r[1])) for r in res)

    def get_all_done_for_file(self, input_file):
        with self.cur() as cur:
            cur.execute('''
                SELECT mut_id FROM results WHERE input_file = ?
            ''', (input_file,))
            res = cur.fetchall()
        return set(r[0] for r in res)

    def mut_is_killed(self, mut_id):
        with self.cur() as cur:
            cur.execute('''
                SELECT 1 FROM results WHERE mut_id = ? AND killed = 1
                LIMIT 1
            ''', (mut_id,))
            res = cur.fetchone()
        return res is not None

    def get_earliest_timestamp_killing_mutant(self, mut_id):
        with self.cur() as cur:
            cur.execute('''
                SELECT timestamp FROM timestamps
                JOIN results ON results.input_file = timestamps.hashname
                WHERE mut_id = ? AND killed = 1
                ORDER BY timestamp
                LIMIT 1
            ''', (mut_id,))
            res = cur.fetchall()
        if len(res) == 0:
            return None
        else:
            return res[0][0]

    def get_timestamp_for_file(self, input_file):
        with self.cur() as cur:
            cur.execute('''
                SELECT timestamp FROM timestamps WHERE hashname = ?
                LIMIT 1
            ''', (input_file,))
            res = cur.fetchall()
        if len(res) == 0:
            return None
        else:
            return res[0][0]

    def add_all_results(self, results):
        with self.cur() as cur:
            cur.execute('''BEGIN IMMEDIATE TRANSACTION''')
            for input_file, mut_id, killed, orig_res, mutant_res in results:
                cur.execute('''
                    INSERT OR IGNORE INTO results (input_file, mut_id, skipped, killed, orig_res, mutant_res)
                        VALUES (?, ?, ?, ?, ?, ?)
                ''', (input_file, mut_id, False, killed, json.dumps(orig_res), json.dumps(mutant_res)))
            cur.execute('COMMIT TRANSACTION')

    def add_all_skipped(self, skipped):
        with self.cur() as cur:
            cur.execute('''BEGIN IMMEDIATE TRANSACTION''')
            for input_file, mut_id in skipped:
                cur.execute('''
                    INSERT OR IGNORE INTO results (input_file, mut_id, skipped, killed, orig_res, mutant_res)
                        VALUES (?, ?, ?, ?, ?, ?)
                ''', (input_file, mut_id, True, None, None, None))
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
    with subprocess.Popen(cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # close_fds=True,
        # errors='backslashreplace',  # text mode: stdout is a str
        # preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
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
    return {
        'killed': original_result['returncode'] != mutant_result['returncode'],
        'original_result': original_result,
        'mutant_result': mutant_result,
    }


def run_inputs(results_db, tmpdir, original_executable, mutants_dir, fuzz_target, chunk):
    results_db = ResultDB(results_db)
    result_counter = Counter()
    skipped = []
    results = []
    for input_file, mut_id in chunk:
        # if results_db.input_on_mut_is_done(Path(input_file).name, mut_id):
        #     result_counter["already_done"] += 1
        #     continue
        # if results_db.mut_is_killed(mut_id):
        mutant_executable = mutants_dir / str(mut_id) / f'{fuzz_target}_{mut_id}'
        res = run_input(
            tmpdir, input_file, original_executable, mutant_executable)
        killed = res['killed']
        orig_res = res['original_result']
        mutant_res = res['mutant_result']
        results.append((Path(input_file).name, mut_id, killed, orig_res, mutant_res))
        result_counter["done"] += 1

    results_db.add_all_skipped(skipped)
    results_db.add_all_results(results)
        #             results_db.add_skipped(input_file, mut_id)
        # results_db.add_result(Path(input_file).name, mut_id, killed, orig_res, mutant_res)
    return result_counter

# for corpus entry in mutants_ids_dir run entry on each covered mutant
# check if corpus entry has already been run, if so skip
# else add corpus entry for each mutant to jobs
def gather_run_jobs_chunk(
    corpus_mutant_id_files, corpus_dir, mutants_dir, fuzz_target, result_db_path
):
    results_db = ResultDB(result_db_path)
    stats = Counter()
    todo_run_jobs = []
    skipped = []
    earliest_kill_cache = {}
    for corpus_mutant_ids_file in corpus_mutant_id_files:
        if corpus_mutant_ids_file.is_dir():
            continue
        input_file = os.path.join(corpus_dir, corpus_mutant_ids_file.name[:-5])

        if not Path(input_file).is_file():
            print(f"corpus mutant ids file {corpus_mutant_ids_file} has no input file {input_file}")
            stats.update(["no_input"])
            continue

        input_file_name = Path(input_file).name

        with open(corpus_mutant_ids_file, 'r') as f:
            try:
                input_mutant_ids = json.load(f)['mut_ids']
            except Exception as e:
                print(f"Error loading json file {corpus_mutant_ids_file}: {e}")
                stats.update(["no_json"])
                continue
        file_timestamp = results_db.get_timestamp_for_file(input_file_name)
        if file_timestamp is None:
            stats.update(["(no_timestamp)"])
            continue
            # print(f"Could not find timestamp for input file: {input_file}")
        all_done_for_file = results_db.get_all_done_for_file(input_file_name)
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
                        skipped.append((input_file_name, mut_id))
                        stats["skipped"] += 1
                        continue
            mutant_executable = mutants_dir / str(mut_id) / f'{fuzz_target}_{mut_id}'
            if not mutant_executable.is_file():
                # print(f"mutant file {mutant_executable} does not exist")  # TODO this is noisy for debug
                stats.update(["no_mutant_executable"])
                continue
            todo_run_jobs.append((input_file, mut_id))
    return todo_run_jobs, stats, skipped


def split_list(lst, num_chunks):
    """Split the list |lst| into |num_chunks| chunks."""
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return chunks(lst, chunk_size)


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
    results_db = ResultDB(result_db_path)
    results_db.initialize(benchmark, fuzz_target, fuzzer, trial_num)
    mutants_dir = shared_mua_binaries_dir / 'mutants' / benchmark

    print(f"\toriginal executable: {original_executable}")
    print(f"\tcorpus dir: {corpus_dir.absolute()}")

    stats = Counter()

    run_jobs_start = time.time()
    num_producer_processes = max(int(os.cpu_count() * 0.1), 1)
    num_runner_threads = os.cpu_count() - num_producer_processes

    mutant_id_files = list(mutants_ids_dir.glob('*'))
    print(f"\t{len(mutant_id_files)=}")
    mutant_id_chunks = chunks(mutant_id_files, ID_CHUNK_SIZE)
    
    # run mutants
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_count = 0
        runner_count = 0
        run_jobs = {}
        todo_list = []
        with ThreadPoolExecutor(max_workers=num_runner_threads) as runner_executor, \
             ProcessPoolExecutor(max_workers=num_producer_processes) as gen_executor:
            while True:
                while gen_count <= num_producer_processes * 2 \
                        and len(todo_list) < (RUN_CHUNK_SIZE * num_runner_threads):
                    try:
                        mut_chunk = next(mutant_id_chunks)
                    except StopIteration:
                        break # all chunks submitted
                    gen_count += 1
                    run_jobs[gen_executor.submit(
                        gather_run_jobs_chunk, mut_chunk, corpus_dir, mutants_dir, fuzz_target, result_db_path
                    )] = "gen"

                if runner_count <= num_runner_threads:
                    if len(todo_list) > RUN_CHUNK_SIZE or gen_count == 0 and len(todo_list) > 0:
                        todo_chunk, todo_list = todo_list[:RUN_CHUNK_SIZE], todo_list[RUN_CHUNK_SIZE:]
                        run_jobs[runner_executor.submit(
                            run_inputs, result_db_path, tmpdir, original_executable, mutants_dir, fuzz_target, todo_chunk
                        )] = "run"
                        runner_count += 1

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
                    print(f"Error while running {job_meta}: {e}:\n{stacktrace}")
                    continue
                if job_meta == "gen":
                    gen_count -= 1
                    assert gen_count >= 0
                    todo_list.extend(res[0])
                    stats.update(res[1])
                    results_db.add_all_skipped(res[2])
                elif job_meta == "run":
                    runner_count -= 1
                    assert runner_count >= 0
                    stats.update(res)

    print(f"\tmua run jobs time: {time.time() - run_jobs_start:0.2f}s")
    print(f"\tmua run stats:")
    for msg, cnt in stats.most_common():
        print(f"\t\t{msg}: {cnt}")
    print(f"\tmua run time: {time.time() - start_time}")
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise e
