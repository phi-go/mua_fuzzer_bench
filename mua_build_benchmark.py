#!/usr/bin/env python3

import json
import os
from pathlib import Path
from pprint import pprint
import sqlite3
import subprocess


def extract_bc(executable_path: Path):
    assert executable_path.is_file(), f"Executable {executable_path} does not exist!"
    print(f"Extracting bitcode from {executable_path}")
    bc_path = executable_path.with_suffix('.bc')
    subprocess.check_call(
         ['get-bc', '-S', '-v', '-o', str(bc_path), str(executable_path)],
         stdout=subprocess.PIPE,
         stderr=subprocess.STDOUT,
         errors='backslashreplace')

    return bc_path


def extract_args(args):
    exception_args = []
    kept_args = []

    for arg in args:
        if not arg.startswith('-'):
            continue

        # Keep these arguments
        if any(arg.startswith(ss) for ss in [
            "-l", "-D", "-I", "-W", "-fno-", "-fPIE", "-pthread", "-stdlib",
            "-std", "-L"
        ]):
            kept_args.append({'val': arg, 'action': None})

        # Ignore these arguments
        elif any(arg.startswith(ss) for ss in [
            "-O", "-o", "-fprofile-instr-generate", "-fcoverage-mapping",
        ]):
            continue

        else:
            exception_args.append(arg)

    if len(exception_args) > 0:
        raise Exception(f"Unhandled args: {exception_args} in {args}")

    return kept_args


def main():
    print("Environment variables:")
    cflags = os.getenv('CFLAGS')
    print(f"CFLAGS: {cflags}")
    cxxflags = os.getenv('CXXFLAGS')
    print(f"CXXFLAGS: {cxxflags}")
    cc = os.getenv('CC')
    print(f"CC: {cc}")
    cxx = os.getenv('CXX')
    print(f"CXX: {cxx}")
    fuzzer_lib = os.getenv('FUZZER_LIB')
    print(f"FUZZER_LIB: {fuzzer_lib}")
    mua_recording_db = os.getenv('MUA_RECORDING_DB')
    print(f"MUA_RECORDING_DB: {mua_recording_db}")
    out = os.getenv('OUT')
    print(f"OUT: {out}")
    src = os.getenv('SRC')
    print(f"SRC: {src}")
    work = os.getenv('WORK')
    print(f"WORK: {work}")
    benchmark = os.getenv('BENCHMARK')
    print(f"BENCHMARK: {benchmark}")
    fuzzer = os.getenv('FUZZER')
    print(f"FUZZER: {fuzzer}")
    print()

    FUZZER_LIB_STR = '/mutator/dockerfiles/programs/common/main.cc'
    CONFIG_PATH = Path('/tmp/config.json')
    recording_db = Path('/tmp/execs.sqlite')

    assert recording_db.is_file(), f"Recording DB {recording_db} does not exist!"

    cmds = []
    with sqlite3.connect(recording_db) as conn:
        c = conn.cursor()
        c.execute("SELECT time, cmd, env FROM cmds")
        for row in c:
            cmds.append(row)

    executables = []
    for _time, cmd_str, env_str in cmds:
            cmd = json.loads(cmd_str)
            env = json.loads(env_str)
            if  FUZZER_LIB_STR in cmd:
                o_args_idx = cmd.index('-o')
                executable_path = Path(cmd[o_args_idx + 1])
                print(f"Found candidate executable: {executable_path}")
                bc_path = extract_bc(executable_path)
                print(f"Extracted bitcode to {bc_path}")
                executables.append((cmd, env, bc_path))

    assert len(executables) > 0, "No bitcode files found!"

    # Create config for each executable bc
    config = {}
    for cmd, _env, bc_path in executables:
        exec_name = Path(bc_path).stem

        is_cpp = None
        if cmd[0].endswith('gclang++'):
            is_cpp = True
        elif cmd[0].endswith('gclang'):
            is_cpp = False
        else:
            raise Exception(f"Unknown executable: {cmd[0]}")

        bin_compile_args = extract_args(cmd)

        exec_config = {
            "bc_compile_args": [],
            "bin_compile_args": bin_compile_args,
            "is_cpp": is_cpp,
            "dict": None,
            "orig_bc": str(bc_path.absolute()),
            "omit_functions": ["main", "LLVMFuzzerTestOneInput"]
        }

        assert exec_name not in config
        config[exec_name] = exec_config

    print(f"Config (written to: {CONFIG_PATH}):")
    pprint(config)

    with open(CONFIG_PATH, 'wt') as f:
        json.dump(config, f, indent=4)

    # Build locator for each bc
    # pipx run hatch run --help

    # cd /mutator && pipx run hatch src/mua_fuzzer_benchmark/eval.py locator_local --config-path /tmp/config.json
    # clear && cd /mutator && gradle build && pipx run hatch run src/mua_fuzzer_benchmark/eval.py locator_local --config-path /tmp/config.json --result-path /tmp/test/

    # /root/go/bin/gclang++ -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION -pthread -Wl,--no-as-needed -Wl,-ldl -Wl,-lm -Wno-unused-command-line-argument -stdlib=libc++ -O3 -I../include /mutator/dockerfiles/programs/common/main.cc ../src/test_lib_json/fuzz.cpp -o /out/jsoncpp_fuzzer lib/libjsoncpp.a

    # /usr/lib/llvm-15/bin/clang++ -fno-inline -O3 -v /out/jsoncpp_fuzzer.ll.opt_mutate.ll /mutator/dockerfiles/programs/common/main.cc -L/mutator/build/install/LLVM_Mutation_Tool/lib -lm -lz -ldl -ldynamiclibrary -o /out/jsoncpp_fuzzer.ll.opt_mutate


if __name__ == "__main__":
    main()
