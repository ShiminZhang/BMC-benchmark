import os
import json
import glob
import random
import multiprocessing
import subprocess
from GenericRA import LOG, LOG_TAG, TOGGLE_SHOWLOG, REG_TAG
from paths import get_cnf_path, get_cnf_per_instance_dir, get_aig_dir, get_solving_log_path, get_solving_times_path, get_solving_times_dir
from utils.utils import run_slurm_job_wrap
from scramble_utils import SCRAMBLE_TYPES, make_scrambled_name, scramble_cnf_file
import argparse
from category import get_all_instance_names

def run_formula(name, K, solver, limit):
    cnf_path = get_cnf_path(name, K)
    solving_log_path = get_solving_log_path(name, K)
    LOG(f"Running formula {name} with K={K} up to limit {limit}")
    if os.path.exists(solving_log_path) and False: # always rerun now
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
    

def generate_cnf(name, K, force=False):
    LOG(f"Generating formula {name} with K={K}")
    cnf_path = get_cnf_path(name, K)
    if os.path.exists(cnf_path) and not force: # TODO check if the cnf is valid
        return True
    cnf_per_instance_dir = get_cnf_per_instance_dir(name)
    aig_dir = get_aig_dir()
    aig_path = f"{aig_dir}/{name}.aig"
    os.system(f"./libs/bin/simplecar -bmc -k {K} -cnf {cnf_per_instance_dir} {aig_path}")
    if os.path.exists(cnf_path):
        return True
    else:
        return False

def restore_cnf_for_name(name):
    TOGGLE_SHOWLOG(True)
    solving_time_path = get_solving_times_path(name)
    if not os.path.exists(solving_time_path):
        LOG(f"solving time json not found for {name}: {solving_time_path}")
        return False
    with open(solving_time_path, "r") as f:
        data = json.load(f)
    ks = sorted([int(k) for k in data.keys()])
    LOG(f"Restoring CNF for {name} with {len(ks)} Ks")
    for k in ks:
        if not generate_cnf(name, k, force=True):
            LOG(f"Failed to regenerate CNF for {name} with K={k}")
            return False
    return True

def generate_and_run_up_to_limit(name, solver, time_limit, k_limit, step = 10):
    LOG(f"Generating and running {name} with K<={k_limit} up to limit {time_limit}")
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
    solver = "./solvers/bin/cadical"
    generate_and_run_up_to_limit(name, solver, time_limit, k_limit, step)

def get_names_with_solving_times():
    solving_times_dir = get_solving_times_dir()
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(solving_times_dir, "*.json"))
    )

def prepare_scrambled_for_name(name, seed, time_limit):
    TOGGLE_SHOWLOG(True)
    solving_time_path = get_solving_times_path(name)
    if not os.path.exists(solving_time_path):
        LOG(f"solving time json not found for {name}: {solving_time_path}")
        return False
    with open(solving_time_path, "r") as f:
        data = json.load(f)
    ks = sorted(int(k) for k in data.keys())
    LOG(f"Scrambling {name} with seed={seed} over {len(ks)} not-timed-out Ks")
    solver = "./solvers/bin/cadical"
    for scramble_type in SCRAMBLE_TYPES:
        scrambled_name = make_scrambled_name(name, scramble_type, seed)
        rng = random.Random(seed)
        for k in ks:
            if not generate_cnf(name, k):
                LOG(f"Failed to generate CNF for {name} with K={k}, skipping scramble")
                continue
            orig_cnf_path = get_cnf_path(name, k)
            # get_cnf_path creates the per-instance dir for scrambled_name as a side effect
            scrambled_cnf_path = get_cnf_path(scrambled_name, k)
            scramble_cnf_file(orig_cnf_path, scrambled_cnf_path, scramble_type, rng)
            run_formula(scrambled_name, k, solver, time_limit)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--time_limit", type=int, default=1600, required=False)
    parser.add_argument("--k_limit", type=int, default=500, required=False)
    parser.add_argument("--step", type=int, default=1, required=False)
    parser.add_argument("--manage", action="store_true", required=False)
    parser.add_argument("--clear", action="store_true", required=False)
    parser.add_argument("--restore_cnf", action="store_true", required=False)
    parser.add_argument("--scramble", type=int, default=None, required=False, help="seed; scrambles every not-timed-out K found in results/solving_times/<name>.json")
    args = parser.parse_args()
    # insufficent_names = ['picorv32_mutCY_nomem-p3', 'oski15a01b70s', '6s207rb28', 'beemlann2f1', 'pdtvisns3p02', 'cal123', '6s0', '6s428rb098', 'oski15a01b51s', 'pdtpmscoherence', 'cal118', 'cal143', 'beemlifts2b1', 'arbitrated_top_n3_w16_d128_e0', 'shift_register_top_w16_d64_e0', '6s13', '6s38', 'beemandrsn4b1', 'oski15a01b41s', 'arbitrated_top_n3_w8_d128_e0', 'oski15a01b03s', 'cal176', '6s404rb4', '6s357r', 'intel066', 'cal149', 'oski15a08b15s', 'oski15a08b03s', 'cal81', 'oski15a08b14s', 'oski15a01b59s', 'cal86', '6s31', 'oski15a01b02s', '6s350rb46', '6s350rb35', 'cal129', 'oski15a08b05s', 'cal34', 'picorv32_mutBY_nomem-p7', '6s320rb0', 'cal102', 'cal106', 'cal112', 'cal33', 'shift_register_top_w16_d16_e0']
    if args.manage:
        log_dir = "./logs/prepare_formulas/"
        os.makedirs(log_dir, exist_ok=True)
        if args.scramble is not None:
            interested_names = get_names_with_solving_times()
            for name in interested_names:
                run_slurm_job_wrap(
                    f"python -m src.scripts.prepare_formulas --name {name} --scramble {args.scramble} --time_limit {args.time_limit}",
                    f"{log_dir}/{name}_scramble_{args.scramble}.log",
                    f"pf_scr_{name}_{args.scramble}", mem="16g", time="20:00:00"
                )
        else:
            interested_names = get_all_instance_names()
            # interested_names = insufficent_names
            for name in interested_names:
                if args.restore_cnf:
                    run_slurm_job_wrap(
                        f"python -m src.scripts.prepare_formulas --name {name} --restore_cnf",
                        f"{log_dir}/{name}_restore_cnf.log",
                        f"pf_restore_{name}", mem="16g", time="5:00:00"
                    )
                else:
                    run_slurm_job_wrap(
                        f"python -m src.scripts.prepare_formulas --name {name} --time_limit {args.time_limit} --k_limit {args.k_limit} --step {args.step}",
                        f"{log_dir}/{name}_{args.k_limit}.log",
                        f"pf_{name}_{args.k_limit}", mem="16g", time="20:00:00"
                    )
    else:
        if args.clear:
            cnf_per_instance_dir = get_cnf_per_instance_dir(args.name)
            for file in os.listdir(cnf_per_instance_dir):
                os.remove(os.path.join(cnf_per_instance_dir, file))
        elif args.scramble is not None:
            prepare_scrambled_for_name(args.name, args.scramble, args.time_limit)
        else:
            if args.restore_cnf:
                restore_cnf_for_name(args.name)
            else:
                prepare_single(args.name, args.time_limit, args.k_limit, args.step)
    # prepare_single("6s0", 1600, 100)

if __name__ == "__main__":
    main()