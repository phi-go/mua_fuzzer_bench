#!/usr/bin/env python3

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os, argparse, random, subprocess, json
from pathlib import Path
from multiprocessing import Pool
import tempfile
import traceback

MAPPED_DIR = Path('/tmp/experiment-data/')

# POOL_SIZE = 10


# for each mutant compilation
# 1. check if mutant is already compiled, if yes, exit
# 2. create lock file in tmp dir
# 3. check if mutant is already compiled, if yes, exit
# 4. if mutant (.tmp) file exists, delete
# 5. (re)create config file
# 6. build mutant to (.tmp)
# 7. rename mutant remove (.tmp)
# 8. delete config file
# 9. delete lock file


def acquireLock(path):
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open(path, 'w+')
    try:
        fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # another instance already locked this
        return None
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()

def build_mutant(mut_id, mutants_dir, fuzz_target, debug_num_mutants):

    if debug_num_mutants is not None:
        num_lock_files = len(list(mutants_dir.glob("*.lock")))
        num_mutant_binaries = len(list(filter(
            lambda ff: ff.suffix != '.lock' and ff.suffix != '.json' and ff.is_file(),
            mutants_dir.glob("*"))
        ))
        if num_lock_files + num_mutant_binaries >= debug_num_mutants:
            return

    # built covered mutant if not build yet
    mutant_lock = mutants_dir/f'{mut_id}.lock'
    lock_fd = acquireLock(mutant_lock)
    if lock_fd is None:
        # lock is already taken
        return

    # if mutant exists, do not build
    mutant_file = mutants_dir/f'{mut_id}'
    config_file = mutant_file.with_suffix(".json")
    
    if(os.path.exists(mutant_file)):
        return
    if(os.path.isfile(config_file)):
        return
    
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
        '/tmp/test/stats.db',
        '--mutation-list',
        config_file
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        env['MUT_SHARED_DIR'] = tmp_dir
        env['MUT_HOST_TMP_PATH'] = tmp_dir
        subprocess.run(
            build_command,
            cwd='/mutator',
            env=env,
        )

    #cleanup
    os.remove(config_file)
    releaseLock(lock_fd)

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
                timeout=10,
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

    fuzz_target = args.fuzz_target
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)
    debug_num_mutants = args.debug_num_mutants

    shared_mua_binaries_dir = MAPPED_DIR / experiment / 'mua-binaries'
    corpus_dir = shared_mua_binaries_dir / 'corpi' / fuzzer / trial_num
    mutants_ids_dir = shared_mua_binaries_dir / 'mutant_ids' / fuzzer / trial_num

    mutants_dir = shared_mua_binaries_dir / 'mutants'
    locator_path = (Path('/out') / fuzz_target).with_suffix('.locator')

    print(f"mutants_ids_dir_entry: {mutants_ids_dir}")

    all_mut_ids = set()

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
            print(f"Completed {completed_count}/{corpus_len} corpus entries, {successful_count} successful, {timeout_count} timed out, {errored_count} errored out")

    print(f"Locator: All completed for {fuzz_target} {experiment} {fuzzer} {trial_num}: {completed_count}/{corpus_len} corpus entries. New results: {successful_count} successful, {timeout_count} timed out, {errored_count} errored out.")

    # build mutants
    build_jobs = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for mut_id in all_mut_ids:
            build_jobs.append(executor.submit(build_mutant, mut_id, mutants_dir, fuzz_target, debug_num_mutants))

    for job in as_completed(build_jobs):
        try:
            job.result()
        except Exception as e:
            stacktrace = traceback.format_exc()
            print(f"Error while building mutant: {e}:\n{stacktrace}")


if __name__ == "__main__":
    main()
