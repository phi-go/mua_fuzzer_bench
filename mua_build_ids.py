#!/usr/bin/env python3

import os, argparse, random, subprocess, json

from multiprocessing import Pool

POOL_SIZE = 10

mutants_dir = ''
fuzz_target = ''

import fcntl

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

def build_mutants(all_ids):

    # shuffle to avoid continously racing builts
    random.shuffle(all_ids)

    # built covered mutant if not build yet

    for id in all_ids:
        mutant_lock = mutants_dir+str(id)+'.lock'
        lock_fd = acquireLock(mutant_lock)
        if lock_fd is None:
            # lock is already taken
            continue

        # if mutant exists, do not build
        mutant_file = mutants_dir+str(id)
        config_file = mutant_file+".json"
        
        if(os.path.exists(mutant_file)):
            continue
        if(os.path.isfile(config_file)):
            continue
        
        # create config file
        mutationlist = list()
        with open(config_file, "w") as f:
            json.dump([{
                "prog": fuzz_target,
                "mutation_ids": [id],
                "mode": "single",
            }], f)
        
        # build new mutant and store it in mutants_dir
        build_command = 'pipx run hatch run src/mua_fuzzer_benchmark/eval.py locator_mutants_local --result-path '+mutant_file+' --statsdb /tmp/test/stats.db --mutation-list '+config_file
        subprocess.run(build_command.split(' '), cwd='/mutator')
        #cleanup
        
        os.remove(config_file)
        releaseLock(lock_fd)


def main():
    global mutants_dir
    global fuzz_target

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

    args = parser.parse_args()

    fuzz_target = args.fuzz_target
    experiment = args.experiment
    fuzzer = args.fuzzer
    trial_num = str(args.trial_num)

    shared_mua_binaries_dir = '/tmp/experiment-data/'+experiment+'/mua-binaries/'
    corpus_dir = shared_mua_binaries_dir+'corpi/'+fuzzer+'/'+trial_num+'/'
    mutants_ids_dir_entry = shared_mua_binaries_dir+'mutant_ids/'+fuzzer+'/'+trial_num+'/'
    ids_dir = mutants_ids_dir_entry+'trigger_signal/'

    mutants_dir = shared_mua_binaries_dir+'mutants/'

    print("mutants_ids_dir_entry:")
    print(mutants_ids_dir_entry)

    # execute corpus with locator
    for corpus_entry in os.listdir(corpus_dir):
        print("corpus_entry")
        print(corpus_entry)
        input_file = os.path.join(corpus_dir, corpus_entry)
        if os.path.isdir(input_file):
            continue
        locator_command = '/out/'+fuzz_target+'.locator '+str(input_file)
        subprocess.run(locator_command.split(' '), cwd=mutants_ids_dir_entry)

    # get list of covered mutant ids
    all_ids = list()

    for id_entry in os.listdir(ids_dir):
        if os.path.isdir(os.path.join(ids_dir, id_entry)):
            continue
        all_ids.append(int(id_entry))

    #TODO: JUST FOR TESTING, REMOVE THE FOLLOWING LINE!!!
    all_ids = all_ids[:20]

    # build mutants
    pool = Pool(POOL_SIZE)
    pool.map(build_mutants, POOL_SIZE*[all_ids])
        

main()