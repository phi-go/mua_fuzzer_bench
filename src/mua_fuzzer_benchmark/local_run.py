

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import logging
import tempfile
import time
from dataclasses import dataclass
from traceback import format_exc
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union
from data_types import CompileArg, LocalProgramConfig, Mutation, MutationLocal, MutationType, RerunMutations, SuperMutant
from constants import EXEC_ID, HOST_TMP_PATH, LOCAL_ROOT_DIR, MUT_BC_SUFFIX, MUT_LL_SUFFIX, NUM_CPUS, SHARED_DIR
from db import ReadStatsDb, Stats
from helpers import mutation_detector_path, mutation_locations_path, CpuCores, RerunMutationsDict, build_compile_args, mutation_prog_source_path, prepare_shared_and_tmp_dir_local, CustomJSONEncoder, should_run, sigint_handler

logger = logging.getLogger(__name__)


@dataclass
class MutationLocalRun:
    mutation_ids: Set[int]
    orig_bc: Path
    bc_args: List[CompileArg]
    bin_args: List[CompileArg]
    is_cpp: bool


def load_local_config(config_path_s: str) -> List[LocalProgramConfig]:
    def load_compile_arg(elem: Any) -> CompileArg:  # type: ignore[misc]
        assert isinstance(elem, dict), f"Expected dict, with keys 'val' and 'action', got {type(elem)}"  # type: ignore[misc]
        assert 'val' in elem.keys(), f"Expected dict, with keys 'val' and 'action', got {elem.keys()}"
        assert 'action' in elem.keys(), f"Expected dict, with keys 'val' and 'action', got {elem.keys()}"
        return CompileArg(elem['val'], elem['action'])


    config_path = Path(config_path_s)
    assert config_path.is_file(), f"Config path {config_path} does not exist or is not a file."

    config = []
    with open(config_path, "rt") as f:
        data_raw = json.load(f) # type: ignore[misc]
        for prog, prog_data in data_raw.items(): # type: ignore[misc]
            config.append(LocalProgramConfig(
                name=prog, # type: ignore[misc]
                bc_compile_args=[load_compile_arg(aa) for aa in prog_data['bc_compile_args']], # type: ignore[misc]
                bin_compile_args=[load_compile_arg(aa) for aa in prog_data['bin_compile_args']], # type: ignore[misc]
                is_cpp=prog_data['is_cpp'], # type: ignore[misc]
                orig_bc=Path(prog_data['orig_bc']).absolute(), # type: ignore[misc]
                omit_functions=prog_data['omit_functions'], # type: ignore[misc]
            ))
    return config


def run_exec_local(  # type: ignore[misc]
        raise_on_error: bool,
        cmd: List[str],
        timeout: Optional[int] = None,
        **kwargs: Any,
) -> Dict[str, Union[int, str, bool]]:
    """
    Run command locally.
    If return_code is not 0, raise a ValueError containing the run result.
    """
    timed_out = False
    # cmd: List[str] = ["docker", "exec", *(exec_args if exec_args is not None else []), container_name, *cmd]
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
            logger.debug(msg)
            raise ValueError(msg)

        return {'returncode': returncode, 'out': stdout, 'timed_out': timed_out}


def instrument_prog_local(prog_config: LocalProgramConfig) -> Dict[str, Union[int, str, bool]]:
    # Compile the mutation location detector for the prog.
    args = ["./run_mutation.py",
            "-bc", str(prog_config.orig_bc),
            *(["-cpp"] if prog_config.is_cpp else ['-cc']),  # specify compiler
            "--bc-args=" + build_compile_args(
                prog_config.bc_compile_args, str(LOCAL_ROOT_DIR)),
            "--bin-args=" + build_compile_args(
                prog_config.bin_compile_args, str(LOCAL_ROOT_DIR))]
    try:
        return run_exec_local(True, args)
    except Exception as e:
        logger.warning(f"Exception during instrumenting {e}")
        raise e


def new_mutation_local(
    mutation_id: int,
    mutation_data: Dict[str, str],
    prog: LocalProgramConfig
) -> MutationLocal:
    return MutationLocal(
        mutation_id=int(mutation_id),
        prog=prog,
        type_id=mutation_data.pop('type'),
        directory=mutation_data.pop('directory'),
        filePath=mutation_data.pop('filePath'),
        line=int(mutation_data.pop('line')),
        column=int(mutation_data.pop('column')),
        instr=mutation_data.pop('instr'),
        funname=mutation_data.pop('funname'),
        additional_info=json.dumps(mutation_data)  # might not all be str
    )


def get_mutation_locator_and_data_local(
    stats: Stats,
    progs: List[LocalProgramConfig],
) -> List[Tuple[str, Path, Path, Path, LocalProgramConfig]]:
    "Get the mutation locator executable and the data for all mutations."

    progs_locator: List[Tuple[str, Path, Path, Path, LocalProgramConfig]] = []

    for prog_config in progs:
        logger.info("="*50)
        logger.info(f"Compiling base and locating mutations for {prog_config.name}")

        _instrument_result = instrument_prog_local(prog_config)

        # get info on mutations
        with open(mutation_locations_path(prog_config), 'rt') as f:
            mutation_data: List[Dict[str, str]] = json.load(f)

        mutations = list(new_mutation_local(int(p['UID']), p, prog_config) for p in mutation_data)

        # Remove mutations for functions that should not be mutated
        omit_functions = prog_config.omit_functions
        mutations = [mm for mm in mutations if mm.funname not in omit_functions]

        bc_path = Path(prog_config.orig_bc)
        detector_path = mutation_detector_path(prog_config)
        mutationlocations_path = mutation_locations_path(prog_config)

        stats.new_prog(EXEC_ID, prog_config.name, prog_config)

        logger.info(f"Found {len(mutations)} mutations for {prog_config.name}")
        for mut in mutations:
            stats.new_mutation(EXEC_ID, mut)

        progs_locator.append((
            prog_config.name,
            bc_path,
            detector_path,
            mutationlocations_path,
            prog_config))

    return progs_locator


def locator_local(
    config_path_s: str,
    result_path_s: str,
) -> None:
    result_path = Path(result_path_s).absolute()
    assert not result_path.exists(), f"Result path {result_path} already exists."

    config = load_local_config(config_path_s)
    assert len(config) > 0, "No programs found in config."
    for prog_config in config:
        assert prog_config.orig_bc.is_file(), f"Orig bc {prog_config.orig_bc} is not a file."

    # change workdir to that of the script
    os.chdir(LOCAL_ROOT_DIR)

    # prepare_mutator_docker_image(fresh_images)
    prepare_shared_and_tmp_dir_local()

    execution_start_time = time.time()

    # prepare environment
    base_shm_dir = SHARED_DIR/"mua_locator"
    base_shm_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the stats object
    tmp_db_path = base_shm_dir/"stats.db"
    stats = Stats(str(tmp_db_path))

    # Record current eval execution data
    # Get the current git status
    logger.warning("WARNING: git status is not recorded for local locator runs.")
    # git_status = get_git_status()
    git_status = "local"
    stats.new_execution(
        EXEC_ID, platform.uname()[1], git_status, None, execution_start_time,
        "",
        json.dumps({k: v for k, v in os.environ.items()}) # type: ignore[misc]
    )

    class MutationDocEntry(TypedDict):
        pattern_name: str
        typeID: int
        pattern_location: str
        pattern_class: str
        description: str
        procedure: str

    # Get a record of all mutation types.
    with open("mutation_doc.json", "rt") as f:
        mutation_types: List[MutationDocEntry] = json.load(f)
        for mt in mutation_types:
            mutation_type = MutationType(
                pattern_name=mt['pattern_name'],
                type_id=mt['typeID'],
                pattern_location=mt['pattern_location'],
                pattern_class=mt['pattern_class'],
                description=mt['description'],
                procedure=mt['procedure'],
            )
            stats.new_mutation_type(mutation_type)

    locators = get_mutation_locator_and_data_local(stats, config)

    stats.init_local_config_table()

    for prog, bc_path, locator, mutationlocations, individual_config in locators:
        result_prog_path = result_path/"progs"/f"{prog}"
        result_prog_path.mkdir(parents=True, exist_ok=False)

        result_locator_path = result_prog_path/f"{prog}.locator"
        shutil.copy(locator, result_locator_path)

        result_locations_path = result_prog_path/f"{prog}.mutationlocations"
        shutil.copy(mutationlocations, result_locations_path)

        result_bc_path = result_prog_path/f"{prog}.bc"
        shutil.copy(bc_path, result_bc_path)

        individual_config_path = result_prog_path/"config.json"
        with open(individual_config_path, "wt") as f:
            json.dump(individual_config, f, indent=4, cls=CustomJSONEncoder)  # type: ignore[misc]

        stats.add_local_config(
            EXEC_ID,
            prog,
            str(result_locator_path.resolve()),
            str(result_locations_path.resolve()),
            str(result_bc_path.resolve()),
            str(individual_config_path.resolve()))

        print(f"files for {prog} copied to {result_locator_path}")

    # Record total time for this execution.
    stats.execution_done(EXEC_ID, time.time() - execution_start_time)

    result_path.mkdir(parents=True, exist_ok=True)
    # Copy the stats db to the result path
    result_db_path = result_path.joinpath("stats.db")
    shutil.copy(tmp_db_path, result_db_path)
    print(f"result db copied to {result_db_path}")


def get_mutation_locator_mutations_local(
    stats: Stats,
    statsdb: ReadStatsDb,
    mutation_list: Dict[str, RerunMutations],
) -> List[MutationLocalRun]:
    config = statsdb.get_local_config()
    progs = list(mutation_list.keys())

    assert all(pp in config for pp in progs), f"Not all programs in mutation list are in stats db: {progs}, {config}"

    all_mutations: List[MutationLocalRun] = []

    for prog in progs:
        locator_path, locations_path, bc_path, prog_config = config[prog]
        start = time.time()
        # load_rerun_prog_local(statsdb, prog)

        # get info on mutations
        with open(locations_path, 'rt') as f:
            mutation_data: List[Dict[str, str]] = json.load(f)

        if len(mutation_data) == 0:
            msg = f"No mutations found for {prog}.\n"
            raise ValueError(msg)

        mutations = list(new_mutation_local(int(p['UID']), p, prog_config) for p in mutation_data)

        # Remove mutations for functions that should not be mutated
        omit_functions = prog_config.omit_functions
        # mutations = [mm for mm in mutations if mm.funname not in omit_functions]

        # If rerun_mutations is specified, collect those mutations
        mutations_dict = {int(mm.mutation_id): mm for mm in mutations}
        rerun_chosen_mutations: List[MutationLocalRun] = []
        rerun_mutations_for_prog = mutation_list[prog].mutation_ids
        for mtu in rerun_mutations_for_prog:
            assert mtu in mutations_dict.keys(), "Can not find specified rerun mutation id in mutations for prog."
            mut = mutations_dict[mtu]
            logger.info(f"Found mutation id: {mut.mutation_id:8<} in function {mut.funname} for {prog}")
            if mut.funname in omit_functions:
                logger.info(f"Skipping mutation id: {mut.mutation_id} because it is in function {mut.funname} which is a omitted function.")
                continue
            # rerun_chosen_mutations.append(mutations_dict[mtu])
            rerun_chosen_mutations.append(MutationLocalRun(
                mutation_ids=set([mtu]),
                orig_bc=bc_path,
                bc_args=prog_config.bc_compile_args,
                bin_args=prog_config.bin_compile_args,
                is_cpp=prog_config.is_cpp
            ))

        logger.info(f"Found {len(rerun_chosen_mutations)} selected mutations for {prog}")

        all_mutations.extend(rerun_chosen_mutations)
        logger.info(f"Preparations for {prog} took: {time.time() - start:.2f} seconds")

    return all_mutations


def build_no_action_compile_args(args: List[CompileArg]) -> str:
    for arg in args:
        assert arg.action is None, f"Expected no action, got {arg}"
    resolved_args = [arg.val for arg in args]
    return " ".join(map(shlex.quote, resolved_args))


@dataclass
class LocalCompileResult:
    mutant_path: Path


def compile_mutation(core: int, mutation: MutationLocalRun, mut_base_dir: Path) -> LocalCompileResult:
    assert len(mutation.mutation_ids) > 0, "No mutations to prepare!"

    bc_args = build_no_action_compile_args(mutation.bc_args)
    bin_args = build_no_action_compile_args(mutation.bin_args)
    # if WITH_ASAN:
    #     compile_args = "-fsanitize=address " + compile_args
    # if WITH_MSAN:
    #     compile_args = "-fsanitize=memory " + compile_args

    prog_bc_name = (Path(mutation.orig_bc).with_suffix(MUT_BC_SUFFIX).name)
    prog_ll_name = (Path(mutation.orig_bc).with_suffix(MUT_LL_SUFFIX).name)

    try:
        cmd = [
            "/mutator/run_mutation.py",
            "-ll", "-bc",
            *(["-cpp"] if mutation.is_cpp else ['-cc']),  # conditionally add cpp flag
            *["-ml", *[str(mid) for mid in mutation.mutation_ids]],
            "--bc-args", str(bc_args),
            "--bin-args", str(bin_args),
            "--out-dir", str(mut_base_dir),
            "--binary",
            str(mutation.orig_bc)
        ]
        run_exec_local(True, cmd, cwd=mut_base_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to compile mutation") from exc

    mutant_path = mut_base_dir/((mutation.orig_bc.with_suffix(".bc.mut")).name)

    return LocalCompileResult(mutant_path)


tasks_type = Dict[Future[LocalCompileResult], Tuple[int, MutationLocalRun, float, Path]]


def start_next_task(
    all_runs: enumerate[MutationLocalRun],
    executor: ThreadPoolExecutor,
    tasks: tasks_type,
    core: int,
) -> Optional[int]:
    # Check if any runs are prepared
    try:
        # Get the next mutant
        ii, mutation = next(all_runs)

    except StopIteration:
        # Done with all mutations and runs, break out of this loop and finish eval.
        return None


    id_str = '_'.join(str(ii) for ii in sorted(mutation.mutation_ids))
    mut_base_dir = Path(tempfile.mkdtemp(
        prefix=f"local_mutation_{id_str}_"))

    logger.debug(f"Starting task for {mutation.mutation_ids} on core {core} in dir {mut_base_dir}")
    tasks[executor.submit(compile_mutation, core, mutation, mut_base_dir)] = \
        (core, mutation, time.time(), mut_base_dir)
    return ii


def wait_for_compile_task(
    stats: Stats,
    tasks: tasks_type,
    cores: CpuCores,
    result_dir: Path,
) -> None:
    "Wait for a task to complete and process the result."
    if len(tasks) == 0:
        logger.info("WARN: Trying to wait for a task but there are none.")
        logger.info(cores.cores)
        return

    # wait for a task to complete
    completed_task = next(as_completed(tasks))
    # get the data associated with the task and remove the task from the list
    task = tasks[completed_task]
    del tasks[completed_task]

    core = task[0]
    mutant = task[1]
    start_time = task[2]
    workdir = task[3]

    # free the core for future use
    cores.release_core(core)

    try:
        result = completed_task.result()
    except Exception as exc:
        logger.error(
            f"Task for {mutant.mutation_ids} on core {core} crashed, took: {time.time() - start_time:.1f}", 
            exc_info=exc)
    else:
        logger.debug(
            f"Task for {mutant.mutation_ids} on core {core} completed, took: {time.time() - start_time:.1f}.")

        # copy the result to the result dir
        id_str = "_".join(str(mid) for mid in sorted(mutant.mutation_ids))
        result_dir = result_dir/f"{mutant.orig_bc.stem}_{id_str}"
        shutil.copy2(result.mutant_path, result_dir)

    shutil.rmtree(workdir)


def locator_mutants_local(
    statsdb_s: str,
    mutation_list_s: str,
    result_path_s: str,
) -> None:
    signal.signal(signal.SIGINT, sigint_handler)
    result_path = Path(result_path_s)
    assert not result_path.exists(), f"Result path {result_path} already exists."
    result_path.mkdir(parents=True, exist_ok=False)

    mutation_list_path = Path(mutation_list_s)
    assert mutation_list_path.is_file(), f"Mutation list path {mutation_list_path} does not exist or is not a file."

    # change workdir to that of the script
    os.chdir(LOCAL_ROOT_DIR)

    prepare_shared_and_tmp_dir_local()

    execution_start_time = time.time()

    # prepare environment
    base_shm_dir = SHARED_DIR/"mua_locator_mutants"
    base_shm_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the stats object
    tmp_db_path = SHARED_DIR/"mua_locator_mutants/stats.db"
    stats = Stats(":memory:")

    statsdb_p = Path(statsdb_s)
    statsdb = ReadStatsDb(statsdb_p)

    with open(mutation_list_path, "rt") as f:
        mutation_list_data: List[RerunMutationsDict] = json.load(f)
        mutation_list = {
            rm['prog']: RerunMutations(
                prog=rm['prog'],
                mutation_ids=rm['mutation_ids'],
                mode=rm['mode'])
            for rm in mutation_list_data
        }
        del mutation_list_data
    
    mutation_runs = get_mutation_locator_mutations_local(
        stats, statsdb, mutation_list)

    cores = CpuCores(NUM_CPUS)

    # for each mutation and for each fuzzer do a run
    with ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
        tasks: tasks_type = {}
        start_time = time.time()
        num_runs = len(mutation_runs)
        all_runs: enumerate[MutationLocalRun] = enumerate(mutation_runs)
        ii: Optional[int]

        while True:
            # Check if a core is free, if so start next task.
            core = cores.try_reserve_core()
            if core is not None and should_run():

                ii = start_next_task(all_runs, executor, tasks, core)
                if ii is not None:
                    logger.info(f"> {ii+1}/{num_runs}")
                # add tasks while there are more free cores
                if cores.has_free():
                    continue

            # If all tasks are done, stop.
            if len(tasks) == 0:
                break
            # All tasks have been added, wait for a task to complete.
            wait_for_compile_task(stats, tasks, cores, result_path)

    logger.info(f"All done, took {time.time() - start_time:.1f} seconds")
