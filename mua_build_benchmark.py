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


def is_full_arg(arg, arg_candidates):
    for arg_candidate in arg_candidates:
        if arg.startswith(arg_candidate):
            return len(arg) > len(arg_candidate)
    raise Exception(f"Just matched the arg: {arg} now not in arg_candidates: {arg_candidates}")


KEEP_VALUE_ARGS = [
    "-D", "-I", "-W", "-l", "-L", "-std", "-fno-", 
]

KEEP_SINGLE_ARGS = [
    "-fPIE", "-pthread",
    "-stdlib", "-fdiagnostics-color",
    "-m64", "-rdynamic",
]

IGNORE_VALUE_ARGS = [
    "-O", "-o", "-g",
]

IGNORE_SINGLE_ARGS = [
    "-fprofile-instr-generate", "-fcoverage-mapping",
]


def extract_args(args):
    print("extracting args:")
    pprint(args)
    exception_args = []
    kept_args = []

    ii = 0
    while ii < len(args):
        arg = args[ii]

        # No need to keep file args
        if not arg.startswith('-'):
            pass

        # Keep these value arguments
        elif any(arg.startswith(ss) for ss in KEEP_VALUE_ARGS):
            kept_args.append({'val': arg, 'action': None})
            if not is_full_arg(arg, KEEP_VALUE_ARGS):
                ii += 1
                arg = args[ii]
                kept_args.append({'val': arg, 'action': None})

        # Keep these single arguments
        elif any(arg.startswith(ss) for ss in KEEP_SINGLE_ARGS):
            kept_args.append({'val': arg, 'action': None})

        # Ignore these value arguments
        elif any(arg.startswith(ss) for ss in IGNORE_VALUE_ARGS):
            if not is_full_arg(arg, IGNORE_VALUE_ARGS):
                ii += 1
            pass
        # Ignore these single arguments
        elif any(arg.startswith(ss) for ss in IGNORE_SINGLE_ARGS):
            pass

        # Unknown argument
        else:
            exception_args.append(arg)

        ii += 1

    if len(exception_args) > 0:
        raise Exception(f"Unhandled args: {exception_args} in {args}")

    print("extracted args:")
    pprint(kept_args)
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

    FUZZER_LIB_STR = ['/mutator/dockerfiles/programs/common/main.cc', f'/{fuzz_target}"', f'"-o", "{fuzz_target}"']
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
            if any(x in cmd_str for x in FUZZER_LIB_STR):
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
