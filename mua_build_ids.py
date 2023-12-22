#!/usr/bin/env python3

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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


MAPPED_DIR = Path('/mapped/')
LOCATOR_TIMEOUT = 30


@dataclass
class CompilationStatus:
    exec_id: int
    done: bool


class CompileDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file, check_same_thread=False)

    def initialize(self):
        print(f"Initializing compile db {self.db_file}")
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS compilations (
                mut_id INTEGER PRIMARY KEY,
                exec_id BLOB,
                done INTEGER
            )
        ''')
        cur.execute('PRAGMA journal_mode=WAL')
        cur.execute('PRAGMA synchronous=NORMAL')
        self.conn.commit()

    def get_compilation_status(self, mut_id):
        cur = self.conn.cursor()
        cur.execute('SELECT exec_id, done FROM compilations WHERE mut_id=?', (mut_id,))
        res = cur.fetchall()
        all = []
        for r in res:
            all.append(CompilationStatus(r[0], r[1]))
        return all

    def get_num_compiled(self, exec_id):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM compilations WHERE done=1 or exec_id=?', (exec_id,))
        return cur.fetchone()[0]

    def set_started_building(self, mut_id, exec_id):
        cur = self.conn.cursor()
        cur.execute('INSERT INTO compilations (mut_id, exec_id, done) VALUES (?, ?, 0)', (mut_id, exec_id))
        self.conn.commit()

    def set_finished_building(self, mut_id):
        cur = self.conn.cursor()
        cur.execute('UPDATE compilations SET done=1 WHERE mut_id=?', (mut_id,))
        self.conn.commit()


def build_mutant(mut_id, exec_id, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants):
    compile_db = CompileDB(compile_db_path)

    if debug_num_mutants is not None:
        num_compiled = compile_db.get_num_compiled(exec_id)
        if num_compiled >= debug_num_mutants:
            return

    mutant_file = mutants_dir/f'{mut_id}'

    # check if mutant is already compiled
    compilation_status = compile_db.get_compilation_status(mut_id)
    for cs in compilation_status:
        if cs.done is True:
            print("mutant is already compiled")
            if not mutant_file.is_file():
                print(f"Error: mutant_file: {mutant_file} does not exist")
            return
        if cs.exec_id == exec_id:
            return

    # mark mutant as being compiled
    compile_db.set_started_building(mut_id, exec_id)

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
        '/mua_build/stats.db',
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
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while building mutant: {e.output}")
            return
        finally:
            #cleanup
            os.remove(config_file)

    # mark mutant as compiled
    compile_db.set_finished_building(mut_id)


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
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)
    debug_num_mutants = args.debug_num_mutants

    shared_mua_binaries_dir = MAPPED_DIR / experiment / 'mua-binaries'
    corpus_dir = shared_mua_binaries_dir / 'corpi' / fuzzer / trial_num
    mutants_ids_dir = shared_mua_binaries_dir / 'mutant_ids' / fuzzer / trial_num

    compile_db_path = shared_mua_binaries_dir / 'compile.sqlite'
    mutants_dir = shared_mua_binaries_dir / 'mutants'
    locator_path = (Path('/out') / fuzz_target).with_suffix('.locator')

    print(f"mutants_ids_dir_entry: {mutants_ids_dir}")

    compile_db = CompileDB(compile_db_path)
    compile_db.initialize()

    all_mut_ids = set()

    locate_start_time = time.time()

    # execute corpus with locator
    corpus_list = os.listdir(corpus_dir)
    corpus_len = len(corpus_list)
    locate_jobs = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for ii, corpus_entry in enumerate(corpus_list):
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
        if completed_count % 100 == 0:
            print(f"Completed {completed_count}/{corpus_len} corpus entries in {time.time() - locate_start_time:.2f}s, {successful_count} successful, {timeout_count} timed out, {errored_count} errored out", flush=True)

    print(f"Locator: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}: {completed_count}/{corpus_len} corpus entries in {time.time() - locate_start_time:.2f}s. New results: {successful_count} successful, {timeout_count} timed out, {errored_count} errored out.")

    exec_id_bytes = exec_id.bytes

    if debug_num_mutants is not None and len(all_mut_ids) > debug_num_mutants:
        print(f"for debugging reducing num of all_mut_ids: {len(all_mut_ids)} to {debug_num_mutants}")
        all_mut_ids = random.sample(list(all_mut_ids), debug_num_mutants)
        print(f"all_mut_ids: {len(all_mut_ids)}")

    build_start_time = time.time()

    # build mutants
    build_jobs = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for mut_id in all_mut_ids:
            build_jobs.append(executor.submit(build_mutant, mut_id, exec_id_bytes, compile_db_path, mutants_dir, fuzz_target, debug_num_mutants))

    for job in as_completed(build_jobs):
        try:
            job.result()
        except Exception as e:
            stacktrace = traceback.format_exc()
            print(f"Error while building mutant: {e}:\n{stacktrace}")

    print(f"Builder: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}: {len(all_mut_ids)} mutants in {time.time() - build_start_time:.2f}s.")


if __name__ == "__main__":
    main()
