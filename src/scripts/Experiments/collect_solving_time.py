from .. import helpers 
from ..paths import get_solving_times_path, get_solving_log_dir, get_cnf_path
import os
import json
import argparse
from ..GenericRA import LOG, TOGGLE_SHOWLOG
from tqdm import tqdm

# this function is supposed to collect the solving time of a formula with all Ks
# formula_dir: the directory of the formula
# time_limit: the time limit of the formula, not used here
# k_limit: the limit of K, not used here
# return: a path to json file, the json file contains the solving time of the formula with all Ks
def collect_solving_time(formula_dir, time_limit=0, k_limit=0):
    # parse basename from formula_dir
    chunks = formula_dir.split("/")
    basename = chunks[-1] or chunks[-2]
    LOG(f"Collecting solving time for {formula_dir}")
    if not basename:
        LOG(f"basename is empty, skipping {formula_dir}")
        return None
        
    # list all files in formula_dir that end with .log
    log_files = [f for f in os.listdir(formula_dir) if f.endswith(".log")]
    
    output_dict = {}
    for log_file in log_files:
        splitted = log_file.split(".")
        if len(splitted) != 3:
            LOG(f"log_file {log_file} is not valid, skipping")
            continue
        name = splitted[0]
        k = int(splitted[1])
        cnf_path = get_cnf_path(name, k)
        with open(cnf_path, "r") as f:
            number_of_lines = len(f.readlines())
            size_of_cnf = number_of_lines - 2
        solving_time = helpers.GetDataFromLog(os.path.join(formula_dir, log_file))
        if solving_time is not None:
            output_dict[k] = {"solving_time": solving_time, "size_of_cnf": size_of_cnf}
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

    args = parser.parse_args()
    if args.all:
        solving_log_dir = get_solving_log_dir()
        for formula_dir in os.listdir(solving_log_dir):
            collect_solving_time(os.path.join(solving_log_dir, formula_dir))
    elif args.all_slurm:
        solving_log_dir = get_solving_log_dir()
        activate_python = "source ../general/bin/activate"
        for formula_dir in os.listdir(solving_log_dir):
            wrap = f"{activate_python} && python -m src.scripts.Experiments.collect_solving_time --formula_dir {os.path.join(solving_log_dir, formula_dir)}"
            os.system(f"sbatch --job-name=collect_solving_time_{formula_dir} --output={os.path.join(solving_log_dir, formula_dir, 'collect_solving_time.log')} --mem=16g --time=4:00:00 --wrap=\"{wrap}\"")
    else:
        collect_solving_time(args.formula_dir)

if __name__ == "__main__":
    main()
    