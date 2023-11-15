#!/usr/bin/env python3

import os, argparse, subprocess, json, shlex
from pathlib import Path

from multiprocessing import Pool

POOL_SIZE = 10

def run(  # type: ignore[misc]
        raise_on_error: bool,
        cmd,
        timeout = None,
        **kwargs,
):
    """
    Run command locally.
    If return_code is not 0, raise a ValueError containing the run result.
    """
    timed_out = False
    with subprocess.Popen(cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        close_fds=True,
        errors='backslashreplace',  # text mode: stdout is a str
        # preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        **kwargs  # type: ignore[misc]
    ) as proc:
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate(timeout=10)
            timed_out = True
        except Exception:
            proc.kill()
            proc.wait()
            raise

        returncode = proc.poll()

        assert isinstance(returncode, int)
        if raise_on_error and returncode != 0:
            msg = f"exec error: {str(proc.args)}\nreturncode: {returncode}\n{stdout}"
            print(msg)
            raise ValueError(msg)

        return {'returncode': returncode, 'out': stdout, 'timed_out': timed_out}

def run_mutants(all_mutants):
    killed_mutants = list()
    surviving_mutants = list()

    for mutant_file in all_mutants:
        corpus = os.listdir(corpus_dir)
        corpus_len = len(corpus)
        print("corpus:")
        print(corpus)
        for corpus_entry in corpus:
            input_file = os.path.join(corpus_dir, corpus_entry)
            run_command = [mutant_file, input_file]
            result = run(False, run_command, timeout=2, cwd='/tmp/')
            if result['returncode'] != 0 or result['timed_out'] is True:
                killed_mutants.append((mutant_file, corpus_entry, result))
                break

        else: # no break
            surviving_mutants.append((mutant_file, corpus_len))

    result_file = mutants_dir+'/'+fuzzer+'_'+trial_num+'.json'
    
    with open(result_file, "w") as f:
            json.dump([{
                "prog": fuzz_target,
                "killed_mutants": [killed_mutants],
                "surviving_mutants": [surviving_mutants]
            }], f)


def main():
    global mutants_dir
    global fuzz_target
    global corpus_dir
    global fuzzer
    global trial_num


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

    fuzz_target = args.fuzz_target
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)

    shared_mua_binaries_dir = '/tmp/experiment-data/'+experiment+'/mua-binaries/'
    corpus_dir = shared_mua_binaries_dir+'corpi/'+fuzzer+'/'+trial_num+'/'
    print("corpus dir:")
    print(Path(corpus_dir).absolute())

    mutants_dir = shared_mua_binaries_dir+'mutants/'

    mutants = list()
    print("mutant files:")

    for entry_file in os.listdir(mutants_dir):
        if os.path.isfile(entry_file):
            continue
        mutant_file = os.path.join(entry_file, fuzz_target+'_'+entry_file)
        mutant_file = os.path.join(mutants_dir, mutant_file)
        print(mutant_file)
        if not os.path.isfile(mutant_file):
            continue

        mutants.append(mutant_file)

    # run mutants
    print("mutants")
    print(mutants)
    run_mutants(mutants)


    # run mutants
    #pool = Pool(POOL_SIZE)
    #pool.map(run_mutants, POOL_SIZE*[all_ids])
        

main()