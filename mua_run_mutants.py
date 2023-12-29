#!/usr/bin/env python3

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, argparse, subprocess, json, shlex
from pathlib import Path

from multiprocessing import Pool
import tempfile
import time
import traceback
import sqlite3

MAPPED_DIR = Path('/mapped/')
RUN_TIMEOUT = 1
RUN_CHUNK_SIZE = 100

class ResultDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        cur = self.conn.cursor()
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
        cur.execute('PRAGMA journal_mode=WAL')
        cur.execute('PRAGMA synchronous=NORMAL')
        self.conn.commit()

    def print_all_results(self):
        cur = self.conn.cursor()
        cur.execute('''
            SELECT * FROM results
        ''')
        for rr in cur.fetchall():
            print(rr)

    def print_all_timestamps(self):
        cur = self.conn.cursor()
        cur.execute('''
            SELECT * FROM timestamps
        ''')
        for rr in cur.fetchall():
            print(rr)

    def get_all_done(self, cur=None):
        if cur is None:
            cur = self.conn.cursor()
        cur.execute('''
            SELECT input_file, mut_id FROM results
        ''', ())
        res = set(hash((file, mut_id)) for file, mut_id in cur.fetchall())
        return res

    def input_on_mut_is_done(self, input_file, mut_id, cur=None):
        if cur is None:
            cur = self.conn.cursor()
        cur.execute('''
            SELECT 1 FROM results WHERE input_file = ? AND mut_id = ?
            LIMIT 1
        ''', (input_file, mut_id))
        res = cur.fetchall()
        return len(res) == 1

    def get_earliest_timestamp_killing_mutant(self, mut_id, cur=None):
        if cur is None:
            cur = self.conn.cursor()
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

    def get_timestamp_for_file(self, input_file, cur=None):
        if cur is None:
            cur = self.conn.cursor()
        cur.execute('''
            SELECT timestamp FROM timestamps WHERE hashname = ?
            LIMIT 1
        ''', (input_file,))
        res = cur.fetchall()
        if len(res) == 0:
            return None
        else:
            return res[0][0]

    def add_result(self, input_file, mut_id, killed, orig_res, mutant_res):
        cur = self.conn.cursor()
        cur.execute('''BEGIN''')
        if self.input_on_mut_is_done(input_file, mut_id, cur):
            self.conn.rollback()
            print(f"input file {input_file} mut_id {mut_id} already done")
            return
        cur.execute('''
            INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)
        ''', (input_file, mut_id, False, killed, json.dumps(orig_res), json.dumps(mutant_res)))
        self.conn.commit()

    def add_skipped(self, input_file, mut_id):
        cur = self.conn.cursor()
        cur.execute('''BEGIN''')
        if self.input_on_mut_is_done(input_file, mut_id, cur):
            self.conn.rollback()
            print(f"input file {input_file} mut_id {mut_id} already done")
            return
        cur.execute('''
            INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)
        ''', (input_file, mut_id, True, None, None, None))
        self.conn.commit()


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
    extra = ''
    with subprocess.Popen(cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # close_fds=True,
        # errors='backslashreplace',  # text mode: stdout is a str
        # preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        **kwargs  # type: ignore[misc]
    ) as proc:
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            timed_out = True
            stdout, _ = proc.communicate(timeout=1)
        except PermissionError:
            for pp in reversed(Path(cmd[0]).parents):
                res = subprocess.run(['ls', '-la', str(pp)], capture_output=True, text=True, stderr=subprocess.STDOUT)
                extra += f"ls -la {pp}:\n{res.stdout}"
            for pp in reversed(Path(cmd[1]).parents):
                res = subprocess.run(['ls', '-la', str(pp)], capture_output=True, text=True, stderr=subprocess.STDOUT)
                extra += f"ls -la {pp}:\n{res.stdout}"
            proc.kill()
            stdout, _ = proc.communicate(timeout=1)
        except:
            proc.kill()
            stdout, _ = proc.communicate(timeout=1)

        returncode = proc.poll()
        runtime = time.time() - start_time

        return {
            'returncode': returncode,
            'out': stdout.decode('utf-8', 'backslashreplace'),
            'timed_out': timed_out,
            'runtime': runtime,
            'extra': extra,
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

# input_file, original_executable, mutant_executable 
def run_inputs(tmpdir, original_executable, chunk):
    results = []
    for input_file, mutant_executable, mut_id in chunk:
        results.append({
            'input_file': input_file,
            'mut_id': mut_id,
            'result': run_input(tmpdir, input_file, original_executable, mutant_executable),
        })
    return results


def main():
    parser = argparse.ArgumentParser(
                    prog='mua_run_mutants',
                    description='Script runs corpus against all covered mutants',
                    epilog='')

    parser.add_argument('fuzz_target', metavar='T',
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
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)
    original_executable = Path(os.environ['OUT']) / fuzz_target

    shared_mua_binaries_dir = MAPPED_DIR / experiment / 'mua-results'
    corpus_dir = shared_mua_binaries_dir / 'corpi' / fuzzer / trial_num
    mutants_ids_dir = shared_mua_binaries_dir / 'mutant_ids' / fuzzer / trial_num
    corpus_run_results = shared_mua_binaries_dir / 'corpus_run_results' / fuzzer / trial_num
    corpus_run_result_db = ResultDB(corpus_run_results / f'results.sqlite')
    mutants_dir = shared_mua_binaries_dir / 'mutants'

    print(f"original executable: {original_executable}")
    print(f"corpus dir: {corpus_dir.absolute()}")

    all_done = corpus_run_result_db.get_all_done()

    stats = Counter()

    mut_id_kill_timer_cache = {}
    todo_run_jobs = []
    # for corpus entry in mutants_ids_dir run entry on each covered mutant
    # check if corpus entry has already been run, if so skip
    # else add corpus entry for each mutant to jobs
    for corpus_mutant_ids_file in mutants_ids_dir.glob('*'):
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
        file_timestamp = corpus_run_result_db.get_timestamp_for_file(input_file_name)
        if file_timestamp is None:
            stats.update(["(no_timestamp)"])
            print(f"Could not find timestamp for input file: {input_file}")
        for mut_id in input_mutant_ids:
            if hash((input_file_name, mut_id)) in all_done:
                # print(f"corpus input file {input_file} has already been run")
                stats.update(["done"])
                continue
            if mut_id in mut_id_kill_timer_cache:
                earliest_kill = mut_id_kill_timer_cache[mut_id]
            else:
                earliest_kill = corpus_run_result_db.get_earliest_timestamp_killing_mutant(mut_id)
                mut_id_kill_timer_cache[mut_id] = earliest_kill
            if earliest_kill is not None and file_timestamp is not None and file_timestamp >= earliest_kill:
                # print(f"corpus input file {input_file} mut_id {mut_id} has already been killed by earlier input")
                corpus_run_result_db.add_skipped(input_file_name, mut_id)
                stats.update(["skipped"])
                continue
            mutant_executable = mutants_dir / str(mut_id) / f'{fuzz_target}_{mut_id}'
            if not mutant_executable.is_file():
                # print(f"mutant file {mutant_executable} does not exist")  # TODO this is noisy for debug
                stats.update(["no_mutant_executable"])
                continue
            todo_run_jobs.append((input_file, mutant_executable, mut_id))


    print(f"run jobs: {len(todo_run_jobs)}")

    # run mutants
    with tempfile.TemporaryDirectory() as tmpdir:
        run_jobs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for chunk in chunks(todo_run_jobs, RUN_CHUNK_SIZE):
                run_jobs.append(executor.submit(
                    run_inputs, tmpdir, original_executable, chunk
                ))

        for job in run_jobs:
            try:
                res = job.result()
            except Exception as e:
                stacktrace = traceback.format_exc()
                print(f"Error while running input chunk: {e}:\n{stacktrace}")
                continue
            for run_result in res:
                input_file = run_result['input_file']
                mut_id = run_result['mut_id']
                exec_result = run_result['result']
                killed = exec_result['killed']
                orig_res = exec_result['original_result']
                mutant_res = exec_result['mutant_result']
                stats.update(["result"])
                corpus_run_result_db.add_result(Path(input_file).name, mut_id, killed, orig_res, mutant_res)

    print(f"mua run stats:")
    for msg, cnt in stats.most_common():
        print(f"{msg}: {cnt}")
    print(f"mua run time: {time.time() - start_time}")
        

if __name__ == "__main__":
    main()
