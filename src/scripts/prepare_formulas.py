import os
import multiprocessing
import subprocess
from .GenericRA import LOG, LOG_TAG, TOGGLE_SHOWLOG, REG_TAG
from .paths import get_cnf_path, get_cnf_per_instance_dir, get_aig_dir, get_solving_log_path
from .utils.utils import run_slurm_job_wrap
import argparse
from .category import get_all_instance_names

def run_formula(name, K, solver, limit):
    cnf_path = get_cnf_path(name, K)
    solving_log_path = get_solving_log_path(name, K)
    LOG(f"Running formula {name} with K={K} up to limit {limit}")
    if os.path.exists(solving_log_path): # TODO check if the log is valid
        return True
    try:
        # Run the solver command with timeout
        result = subprocess.run(
            f"{solver} {cnf_path} --plain -t {limit}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=limit
        )
        
        # Write output to log file
        with open(solving_log_path, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nSTDERR:\n{result.stderr}")
        
        return True
    except subprocess.TimeoutExpired:
        LOG(f"Formula {name} with K={K} timed out after {limit} seconds")
        return False
    except Exception as e:
        LOG(f"Error running formula {name} with K={K}: {e}")
        return False
    

def generate_cnf(name, K):
    LOG(f"Generating formula {name} with K={K}")
    cnf_path = get_cnf_path(name, K)
    if os.path.exists(cnf_path): # TODO check if the cnf is valid
        return True
    cnf_per_instance_dir = get_cnf_per_instance_dir(name)
    aig_dir = get_aig_dir()
    aig_path = f"{aig_dir}/{name}.aig"
    os.system(f"./libs/bin/simplecar -bmc -k {K} -cnf {cnf_per_instance_dir} {aig_path}")
    if os.path.exists(cnf_path):
        return True
    else:
        return False

def generate_and_run_up_to_limit(name, solver, time_limit, k_limit, step = 10):
    LOG(f"Generating and running {name} with K<={k_limit} up to limit {time_limit}")
    step = 10
    # cnf_per_instance_dir = get_cnf_per_instance_dir(name)
    # generate cnf
    for i in range(1, k_limit + 1, step):
        if generate_cnf(name, i):
            if not run_formula(name, i, solver, time_limit):
                LOG(f"Failed to run formula {name} with K={i}, halting")
                break
        else:
            LOG(f"Failed to generate CNF for {name} with K={i}, halting")
            return False
    return True

def prepare_single(name, time_limit, k_limit, step = 10):
    TOGGLE_SHOWLOG(True)
    time_limit = 1800
    solver = "./solvers/bin/cadical"
    generate_and_run_up_to_limit(name, solver, time_limit, k_limit, step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--time_limit", type=int, default=1600, required=False)
    parser.add_argument("--k_limit", type=int, default=5000, required=False)
    parser.add_argument("--step", type=int, default=10, required=False)
    parser.add_argument("--manage", action="store_true", required=False)
    parser.add_argument("--clear", action="store_true", required=False)
    args = parser.parse_args()

    if args.manage:
        interested_names = get_all_instance_names()
        log_dir = "./logs/prepare_formulas/"
        os.makedirs(log_dir, exist_ok=True)
        for name in interested_names:
            run_slurm_job_wrap(
                f"python -m src.scripts.prepare_formulas --name {name} --time_limit {args.time_limit} --k_limit {args.k_limit}",
                f"{log_dir}/{name}_{args.k_limit}.log",
                f"pf_{name}_{args.k_limit}", mem="16g", time="20:00:00"
                )
    else:
        if args.clear:
            cnf_per_instance_dir = get_cnf_per_instance_dir(args.name)
            for file in os.listdir(cnf_per_instance_dir):
                os.remove(os.path.join(cnf_per_instance_dir, file))
        else:
            prepare_single(args.name, args.time_limit, args.k_limit, args.step)
    # prepare_single("6s0", 1600, 100)

if __name__ == "__main__":
    main()