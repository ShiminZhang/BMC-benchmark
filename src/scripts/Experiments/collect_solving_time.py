from paths import get_solving_times_path, get_solving_log_dir, get_cnf_path, get_cnf_per_instance_dir, get_aig_dir
import os
import json
import argparse
from GenericRA import LOG, TOGGLE_SHOWLOG, GetDataFromLog
from tqdm import tqdm

# this function is supposed to collect the solving time of a formula with all Ks
# formula_dir: the directory of the formula
# time_limit: the time limit of the formula, not used here
# k_limit: the limit of K, not used here
# return: a path to json file, the json file contains the solving time of the formula with all Ks
def get_nvar_from_cnf(cnf_path):
    with open(cnf_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("c"):
                continue
            if stripped.startswith("p"):
                parts = stripped.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    return int(parts[2])
                if len(parts) >= 3:
                    return int(parts[2])
    return None

def restore_cnf_if_needed(name, k, cnf_path, restore_cnf):
    if os.path.exists(cnf_path):
        return True
    if not restore_cnf:
        return False
    LOG(f"Restoring CNF for {name} with K={k}")
    cnf_per_instance_dir = get_cnf_per_instance_dir(name)
    aig_dir = get_aig_dir()
    aig_path = f"{aig_dir}/{name}.aig"
    os.system(f"./libs/bin/simplecar -bmc -k {k} -cnf {cnf_per_instance_dir} {aig_path}")
    return os.path.exists(cnf_path)

def collect_solving_time(formula_dir, time_limit=0, k_limit=0, include_nvar=False, restore_cnf=False):
    # parse basename from formula_dir
    chunks = formula_dir.split("/")
    basename = chunks[-1] or chunks[-2]
    LOG(f"Collecting solving time for {formula_dir}")
    if not basename:
        LOG(f"basename is empty, skipping {formula_dir}")
        return None
        
    # list all files in formula_dir that end with .log
    log_files = [f for f in os.listdir(formula_dir) if f.endswith(".log")]
    output_path = get_solving_times_path(basename)
    k_to_name = {}
    for log_file in log_files:
        splitted = log_file.split(".")
        if len(splitted) == 3:
            try:
                k_to_name[int(splitted[1])] = splitted[0]
            except ValueError:
                continue

    if include_nvar and os.path.exists(output_path):
        with open(output_path, "r") as f:
            output_dict = json.load(f)
        for k_str, entry in output_dict.items():
            k = int(k_str)
            name = k_to_name.get(k, basename)
            cnf_path = get_cnf_path(name, k)
            if restore_cnf_if_needed(name, k, cnf_path, restore_cnf):
                entry["nvar"] = get_nvar_from_cnf(cnf_path)
            else:
                LOG(f"cnf file {cnf_path} not found, skipping nvar for k={k}")
        output_dict = dict(sorted(output_dict.items(), key=lambda x: int(x[0])))
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)
        return output_path
    
    output_dict = {}
    for log_file in log_files:
        splitted = log_file.split(".")
        if len(splitted) != 3:
            LOG(f"log_file {log_file} is not valid, skipping")
            continue
        name = splitted[0]
        k = int(splitted[1])
        cnf_path = get_cnf_path(name, k)
        if not restore_cnf_if_needed(name, k, cnf_path, include_nvar and restore_cnf):
            # try load from previous result
            previous_result_path = get_solving_times_path(basename)
            assert os.path.exists(previous_result_path), f"previous result file {previous_result_path} not found"
            with open(previous_result_path, "r") as f:
                data = json.load(f)
                if str(k) in data:
                    size_of_cnf = data[str(k)]["size_of_cnf"]
                else:
                    LOG(f"k {k} not found in previous result, skipping")
                    continue
        else: 
            with open(cnf_path, "r") as f:
                number_of_lines = len(f.readlines())
                size_of_cnf = number_of_lines - 2
        solving_time, is_unsat = GetDataFromLog(os.path.join(formula_dir, log_file))
        if solving_time is not None:
            record = {"solving_time": solving_time, "size_of_cnf": size_of_cnf, "is_unsat": is_unsat}
            if include_nvar and os.path.exists(cnf_path):
                record["nvar"] = get_nvar_from_cnf(cnf_path)
            output_dict[str(k)] = record
    output_dict = dict(sorted(output_dict.items(), key=lambda x: int(x[0])))

    # save the output_dict to a json file
    with open(get_solving_times_path(basename), "w") as f:
        json.dump(output_dict, f, indent=4)
    # LOG(f"Highest solving time: {max(output_dict.values())}")
    return get_solving_times_path(basename)

def main():
    TOGGLE_SHOWLOG(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula_dir", type=str, required=False)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--all_slurm", action="store_true", default=False)
    parser.add_argument("--n", action="store_true", default=False)
    parser.add_argument("--restore_cnf", action="store_true", default=False)

    args = parser.parse_args()
    if args.all:
        solving_log_dir = get_solving_log_dir()
        for formula_dir in tqdm(os.listdir(solving_log_dir)):
            collect_solving_time(
                os.path.join(solving_log_dir, formula_dir),
                include_nvar=args.n,
                restore_cnf=args.restore_cnf,
            )
    elif args.all_slurm:
        solving_log_dir = get_solving_log_dir()
        activate_python = "source ./.env; source $PYENVPATH"
        for formula_dir in os.listdir(solving_log_dir):
            n_flag = " --n" if args.n else ""
            restore_flag = " --restore_cnf" if args.restore_cnf else ""
            wrap = f"{activate_python} && python -m src.scripts.Experiments.collect_solving_time --formula_dir {os.path.join(solving_log_dir, formula_dir)}{n_flag}{restore_flag}"
            os.system(f"sbatch --job-name=collect_solving_time_{formula_dir} --output={os.path.join(solving_log_dir, formula_dir, 'collect_solving_time.log')} --mem=16g --time=4:00:00 --wrap=\"{wrap}\"")
    else:
        collect_solving_time(
            args.formula_dir,
            include_nvar=args.n,
            restore_cnf=args.restore_cnf,
        )

if __name__ == "__main__":
    main()
    