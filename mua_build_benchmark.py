#!/usr/bin/env python3

import json
import os
from pathlib import Path
from pprint import pprint
import sqlite3
import subprocess
from typing import Optional


def cwd_path(path: Path, cwd: Optional[Path] = None):
    current_dir = Path.cwd()
    if cwd is None:
        return (current_dir / path).absolute()
    else:
        return (cwd / path).absolute()


def extract_bc(executable_path: Path, workdir: str = None, outdir: str = None):
    print(f"Trying to extracting bitcode from {executable_path}")
    bc_path = executable_path.with_suffix('.bc')
    getbc_workdir_error = None
    getbc_outdir_error = None
    try:
        subprocess.check_output(
            ['get-bc', '-S', '-v', '-o', str(bc_path), str(executable_path)],
            stderr=subprocess.STDOUT,
            errors='backslashreplace',
            cwd=outdir,
        )
        return cwd_path(bc_path, outdir)
    except subprocess.CalledProcessError as e:
        error = f"get-bc outdir (OUT: {outdir}) failed\n"
        error += f"get-bc outdir failed: {e}\n"
        error += f"get-bc outdir stdout: {e.stdout}\n"
        error += f"get-bc outdir stderr: {e.stderr}"
        getbc_outdir_error = error
    try:
        subprocess.check_output(
            ['get-bc', '-S', '-v', '-o', str(bc_path), str(executable_path)],
            stderr=subprocess.STDOUT,
            errors='backslashreplace',
            cwd=workdir,
        )
        return cwd_path(bc_path, workdir)
    except subprocess.CalledProcessError as e:
        error = f"get-bc workdir (workdir: {workdir}) failed\n"
        error += f"get-bc workdir failed: {e}\n"
        error += f"get-bc workdir stdout: {e.stdout}\n"
        error += f"get-bc workdir stderr: {e.stderr}"
        getbc_workdir_error = error

    raise Exception(
        f"Failed to extract bitcode from {executable_path}\n" +
        f"{getbc_workdir_error}\n" +
        f"{getbc_outdir_error}"
    )
        



def extract_args(args):
    exception_args = []
    kept_args = []

    for arg in args:
        if not arg.startswith('-'):
            continue

        # Keep these arguments
        if any(arg.startswith(ss) for ss in [
            "-l", "-D", "-I", "-W", "-fno-", "-fPIE", "-pthread", "-stdlib",
            "-std", "-L", "-fdiagnostics-color"
        ]):
            kept_args.append({'val': arg, 'action': None})

        # Ignore these arguments
        elif any(arg.startswith(ss) for ss in [
            "-O", "-o", "-fprofile-instr-generate", "-fcoverage-mapping",
            "-g",
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
    fuzz_target = os.getenv('FUZZ_TARGET')
    print(f"FUZZ_TARGET: {fuzz_target}")
    print()

    FUZZER_LIB_STR = '/mutator/dockerfiles/programs/common/main.cc'
    CONFIG_PATH = Path('/mua_build/config.json')
    recording_db = Path('/mua_build/execs.sqlite')

    assert recording_db.is_file(), f"Recording DB {recording_db} does not exist!"

    cmds = []
    with sqlite3.connect(recording_db) as conn:
        c = conn.cursor()
        c.execute("SELECT time, cmd, env FROM cmds")
        for row in c:
            cmds.append(row)

    candidates = []
    for _time, cmd_str, env_str in cmds:
            cmd = json.loads(cmd_str)
            env = json.loads(env_str)
            if  FUZZER_LIB_STR in cmd:
                o_args_idx = cmd.index('-o')
                executable_path = Path(cmd[o_args_idx + 1])
                print(f"Found candidate executable: {executable_path}")
                candidates.append((cmd, env, executable_path))
    executables = []
    for cmd, env, executable_path in candidates:
        if fuzz_target is not None:
            if fuzz_target != executable_path.name:
                print(f"Skipping candidate executable ({executable_path}) " +
                      f"because it does not match FUZZ_TARGET: {fuzz_target}")
                continue
        print(f"Checking candidate executable: {executable_path}")
        print(f"cmd: {cmd}")
        print(f"env: {env}")
        bc_path = extract_bc(executable_path, env.get('PWD', None), out)
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


if __name__ == "__main__":
    main()
