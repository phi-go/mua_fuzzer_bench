#!/usr/bin/env python3
"""
A python script for orchestrating the mutation of subjects.
"""
import sys
import subprocess
import argparse
import shlex


def run(args):
    print(args, flush=True)
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout.decode(), flush=True)
    return proc


def run_mutation(args):
    """
    Takes the program as argument, checks if it is in a compilable form, converts it to an .ll file if necessary and
    then performs the actual mutation.
    :param prog:
    :return:
    """
    print("run mutation args:", args)

    prog = args.program

    bc_args = shlex.split(args.bc_args)

    if prog.endswith(".bc"):
        # only run the pre computation algorithm if no mutation should be done
        if args.mutate == -2:
            proc_res = run(["clang", "-S", "-emit-llvm", *bc_args, prog, "-o", f"{prog[:-3]}.ll"])
            #  if proc_res != 0:
            #      sys.exit(proc_res.returncode)
        mutate = f"{prog[:-3]}.ll"
    else:
        mutate = prog

    # only run the find algorithm if no mutation should be done
    if args.mutate == -2:
        arguments = ["python3", "build/install/LLVM_Mutation_Tool/bin/compileAndFind.py", mutate]
        if args.cpp:
            arguments.append("-cpp")
        arguments.append("--bc-args=" + args.bc_args)
        arguments.append("--bin-args=" + args.bin_args)
        proc_res = run(arguments)

    # only mutate if a specific mutation id is given
    else:
        arguments = ["python3", "build/install/LLVM_Mutation_Tool/bin/Mutate.py"]
        if args.bitcode:
            arguments.append("-bc")
        if args.bitcode_human_readable:
            arguments.append("-ll")
        if args.binary:
            arguments.append("-bn")
        if args.cpp:
            arguments.append("-cpp")
        arguments.append("-m")
        arguments.append(str(args.mutate))  # type must be string when running the subprocess
        arguments.append(mutate)
        arguments.append("--bc-args=" + args.bc_args)
        arguments.append("--bin-args=" + args.bin_args)
        proc_res = run(arguments)
    sys.exit(proc_res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Script to find and mutate patterns. \
            Need at least one of the arguments [-bc, -ll, -bn] to get resulting files.")
    parser.add_argument('-bc', "--bitcode", action='store_true',
                        help="Keeps the mutated bitcode files.")
    parser.add_argument('-ll', "--bitcode_human_readable", action='store_true',
                        help="Keeps the mutated bitcode files as human readable files.")
    parser.add_argument('-bn', "--binary", action='store_true',
                        help="Creates mutated runnable binaries.")
    parser.add_argument('-cpp', "--cpp", action='store_true',
                        help="Uses clang++ instead of clang for compilation.")
    parser.add_argument("-m", "--mutate", type=int, default=-2,
                        help="Defines which mutation should be applied, -1 if all should be applied.")
    parser.add_argument("--bc-args", default="",
                        help="Compiler arguments that should be used for compilation for all artifacts.")
    parser.add_argument("--bin-args", default="",
                        help="Compiler arguments that should be used for compilation of the binary.")
    parser.add_argument("program",
                        help="Path to the source file that will be mutated.")

    args = parser.parse_args(sys.argv[1:])
    if args.mutate != -2 and not any([args.bitcode, args.bitcode_human_readable, args.binary]):
        parser.error('Need at least one of the arguments [-bc, -ll, -bn] to get resulting files.')

    run_mutation(args)

if __name__ == "__main__":
    main()

