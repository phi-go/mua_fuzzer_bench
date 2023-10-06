import hashlib
import json
import logging
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import is_dataclass, asdict
from types import FrameType
from typing import Dict, List, Optional, TypedDict

from data_types import CompileArg, Fuzzer, LocalProgramConfig, Program

from constants import BLOCK_SIZE, HOST_TMP_PATH, IN_DOCKER_SHARED_DIR, SHARED_DIR

logger = logging.getLogger(__name__)




# Indicates if the evaluation should continue, is mainly used to shut down
# after a keyboard interrupt by the user.
# Global variable that is only written in the sigint_handler, as such it is safe
# to use in a read only fashion by the threads.
should_run_p = True


def should_run() -> bool:
    global should_run_p
    return should_run_p


# Handler for a keyboard interrupt only sets `should_run` to False.
def sigint_handler(signum: int, _frame: Optional[FrameType]) -> None:
    global should_run_p
    logger.info(f"Got stop signal: ({signum}), stopping!")
    should_run_p = False


def fuzzer_container_tag(name: str) -> str:
    return f"mutation-testing-fuzzer-{name}"


def subject_container_tag(name: str) -> str:
    return f"mutation-testing-subject-{name}"


def mutation_locations_path(prog_info: Program | LocalProgramConfig) -> Path:
    orig_bc = Path(prog_info.orig_bc)
    return orig_bc.with_suffix('.bc.mutationlocations')


def mutation_locations_graph_path(prog_info: Program) -> Path:
    orig_bc = Path(prog_info.orig_bc)
    return orig_bc.with_suffix('.bc.mutationlocations.graph')


def mutation_detector_path(prog_info: Program | LocalProgramConfig) -> Path:
    orig_bc = Path(prog_info.orig_bc)
    return  orig_bc.with_suffix(".bc.opt_mutate")


def mutation_prog_source_path(prog_info: Program | LocalProgramConfig) -> Path:
    orig_bc = Path(prog_info.orig_bc)
    return orig_bc.with_suffix('.bc.ll')


def hash_file(file_path: Path) -> str:
    h = hashlib.sha512()
    b  = bytearray(BLOCK_SIZE)
    mv = memoryview(b)
    with open(file_path, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def shared_dir_to_docker(dir: Path) -> Path:
    rel_path = dir.relative_to(SHARED_DIR)
    res = IN_DOCKER_SHARED_DIR/rel_path
    return res


def get_seed_dir(seed_base_dir: Path, prog: str, fuzzer: str) -> Path:
    """
    Gets the seed dir inside of seed_base_dir based on the program name.
    Further if there is a directory inside with the name of the fuzzer, that dir is used as the seed dir.
    Example:
    As a sanity check if seed_base_dir/<prog> contains files and directories then an error is thrown.
    seed_base_dir/<prog>/<fuzzer> exists then this dir is taken as the seed dir.
    seed_base_dir/<prog> contains only files, then this dir is the seed dir.
    """
    prog_seed_dir = seed_base_dir/prog
    seed_paths = list(prog_seed_dir.glob("*"))
    has_files = any(sp.is_file() for sp in seed_paths)
    has_dirs = any(sp.is_dir() for sp in seed_paths)
    if has_files and has_dirs:
        raise ValueError(f"There are files and directories in {prog_seed_dir}, either the dir only contains files, "
              f"in which case all files are used as seeds for every fuzzer, or it contains only directories. "
              f"In the second case the content of each fuzzer directory is used as the seeds for the respective fuzzer.")

    if has_dirs:
        # If the fuzzer specific seed dir exists, return it.
        prog_fuzzer_seed_dir = prog_seed_dir/fuzzer
        if not prog_fuzzer_seed_dir.is_dir():
            raise ValueError(
                f"Expected seed dir to exist: {prog_fuzzer_seed_dir}.")
        return prog_fuzzer_seed_dir

    elif has_files:
        # Else just return the prog seed dir.
        return prog_seed_dir

    # Has no content
    else:
        raise ValueError(f"Seed dir has not content. {prog_seed_dir}")


class CoveredFile:
    def __init__(self, workdir: Path, start_time: float) -> None:
        super().__init__()
        self.found: Dict[int, float] = {}
        self.host_path = SHARED_DIR/"covered"/workdir
        self.host_path.mkdir(parents=True)
        self.docker_path = IN_DOCKER_SHARED_DIR/"covered"/workdir
        self.start_time = start_time

    def check(self) -> Dict[int, float]:
        cur_time = time.time() - self.start_time
        cur = set(int(cf.stem) for cf in self.host_path.glob("*"))
        new_keys = cur - self.found.keys()
        new = {nn: cur_time for nn in new_keys}
        self.found = {**self.found, **new}
        return new

    # def file_path(self):
    #     return self.path

    def __del__(self) -> None:
        shutil.rmtree(self.host_path)


def load_fuzzers() -> Dict[str, Fuzzer]:

    class FuzzerConfig(TypedDict):
        queue_dir: str
        queue_ignore_files: list[str]
        crash_dir: str
        crash_ignore_files: list[str]

    fuzzers = {}
    for fuzzer_dir in Path("dockerfiles/fuzzers").iterdir():
        if fuzzer_dir.name.startswith("."):
            continue # skip hidden files

        if fuzzer_dir.name == "system":
            continue

        if not fuzzer_dir.is_dir():
            continue
        
        fuzzer_config_path = fuzzer_dir/"config.json"
        with open(fuzzer_config_path, "r") as f:
            fuzzer_config: FuzzerConfig = json.load(f)

        fuzzer_name = fuzzer_dir.name

        fuzzers[fuzzer_name] = Fuzzer(
            name=fuzzer_name,
            queue_dir=fuzzer_config['queue_dir'],
            queue_ignore_files=fuzzer_config['queue_ignore_files'],
            crash_dir=fuzzer_config['crash_dir'],
            crash_ignore_files=fuzzer_config['crash_ignore_files'],
        )

    return fuzzers


def load_programs() -> Dict[str, Program]:

    class ProgramConfigArg(TypedDict):
        val: str
        action: Optional[str]

    class ProgramConfig(TypedDict):
        bc_compile_args: List[ProgramConfigArg]
        bin_compile_args: List[ProgramConfigArg]
        dict: str
        is_cpp: bool
        orig_bin: str
        orig_bc: str
        omit_functions: List[str]


    programs = {}
    for prog_dir in Path("dockerfiles/programs").iterdir():
        prog_dir_name = prog_dir.name
        if prog_dir_name.startswith("."):
            continue # skip hidden files

        if prog_dir_name == "common":
            continue

        if not prog_dir.is_dir():
            continue
        
        prog_config_path = prog_dir/"config.json"
        with open(prog_config_path, "r") as f:
            prog_config: Dict[str, ProgramConfig] = json.load(f)

        for prog_config_name, prog_config_data in prog_config.items():

            prog_name = f"{prog_dir_name}_{prog_config_name}"

            assert prog_name not in programs

            try:
                bc_compile_args = [
                    CompileArg(arg['val'], arg['action'])
                    for arg in prog_config_data["bc_compile_args"]
                ]

                bin_compile_args = [
                    CompileArg(arg['val'], arg['action'])
                    for arg in prog_config_data["bin_compile_args"]
                ]

                dict_path_str = prog_config_data["dict"]
                if dict_path_str is not None:
                    dict_path = Path("tmp/programs")/prog_dir_name/dict_path_str
                else:
                    dict_path = None

                programs[prog_name] = Program(
                    name=prog_name,
                    bc_compile_args=bc_compile_args,
                    bin_compile_args=bin_compile_args,
                    is_cpp=prog_config_data["is_cpp"],
                    dict_path=dict_path,
                    orig_bin=Path("tmp/programs")/prog_dir_name/prog_config_data["orig_bin"],
                    orig_bc=Path("tmp/programs")/prog_dir_name/prog_config_data["orig_bc"],
                    omit_functions=prog_config_data["omit_functions"],
                    dir_name=prog_dir_name,
                )
            except KeyError as e:
                raise KeyError(f"Key {e} not found in {prog_config_path}")

    return programs


class CpuCores():
    def __init__(self, num_cores: int):
        self.cores: list[bool] = [False]*num_cores

    def try_reserve_core(self) -> Optional[int]:
        try:
            idx = self.cores.index(False)
            self.cores[idx] = True
            return idx
        except ValueError:
            return None

    def release_core(self, idx: int) -> None:
        assert self.cores[idx] is True, "Trying to release an already free core"
        self.cores[idx] = False

    def has_free(self) -> bool:
        return any(cc is False for cc in self.cores)

    def usage(self) -> float:
        return len([cc for cc in self.cores if cc]) / len(self.cores)


class RerunMutationsDict(TypedDict):
    prog: str
    mutation_ids: List[int]
    mode: str


def resolve_compile_args(args: List[CompileArg], workdir: str) -> List[str]:
    resolved = []
    for arg in args:
        if arg.action is None:
            resolved.append(arg.val)
        elif arg.action == 'prefix_workdir':
            resolved.append(str(Path(workdir)/arg.val))
        else:
            raise ValueError("Unknown action: {}", arg)
    return resolved


def build_compile_args(args: List[CompileArg], workdir: str) -> str:
    resolved_args = resolve_compile_args(args, workdir)
    return " ".join(map(shlex.quote, resolved_args))


def prepare_shared_and_tmp_dir_local() -> None:
    # SHARED_DIR
    if SHARED_DIR.exists():
        if SHARED_DIR.is_dir():
            logger.info(f"Cleaning up already existing shared dir: {SHARED_DIR}.")
            try:
                shutil.rmtree(SHARED_DIR)
            except OSError as err:
                logger.info(f"Could not clean up {SHARED_DIR}: {err}")
        if SHARED_DIR.is_file():
            raise Exception(f"The specified location for shared dir is a file: {SHARED_DIR}.")

    SHARED_DIR.mkdir(parents=True)

    # ./tmp
    for td in ['lib', 'programs', 'unsolved_mutants']:
        tmp_dir = HOST_TMP_PATH/td
        if tmp_dir.exists():
            if tmp_dir.is_dir():
                logger.info(f"Cleaning up already existing tmp dir: {tmp_dir}.")
                try:
                    shutil.rmtree(tmp_dir)
                except OSError as err:
                    logger.info(f"Could not clean up {tmp_dir}: {err}")
            if tmp_dir.is_file():
                raise Exception(f"The specified location for tmp dir is a file: {tmp_dir}.")

        tmp_dir.mkdir(parents=True)


def prepare_shared_dir_and_tmp_dir() -> None:
    """
    Prepare the shared dir and ./tmp dir for the evaluation containers.
    The shared dir will be deleted if it already exists.
    """
    
    prepare_shared_and_tmp_dir_local()

    proc = subprocess.run(f"""
        docker rm dummy || true
        docker create -ti --name dummy mutator_mutator bash
        docker cp dummy:/home/mutator/programs/common/ tmp/programs/common/
        docker cp dummy:/home/mutator/build/install/LLVM_Mutation_Tool/lib/ tmp/
        docker rm -f dummy
    """, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        logger.info(f"Could not extract mutator files.", proc)
        sys.exit(1)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o): # type: ignore
        if is_dataclass(o): # type: ignore[misc]
            return asdict(o) # type: ignore[misc]
        if isinstance(o, Path):
            return str(o)
        return super().default(o)
