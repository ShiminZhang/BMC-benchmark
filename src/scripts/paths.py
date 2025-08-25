import os

def get_cnf_dir(K):
    if os.path.exists(f"./data/cnfs/"):
        return f"./data/cnfs/"
    else:
        os.makedirs(f"./data/cnfs/")
        return f"./data/cnfs/"

def get_cnf_per_instance_dir(name, K):
    cnf_dir = get_cnf_dir(K)
    cnf_per_intance_dir = f"{cnf_dir}/{name}"
    if not os.path.exists(cnf_per_intance_dir):
        os.makedirs(cnf_per_intance_dir)
    return cnf_per_intance_dir

def get_cnf_path(name, K):
    cnf_per_intance_dir = get_cnf_per_instance_dir(name, K)
    return f"{cnf_per_intance_dir}/{name}.{K}.cnf"

def get_solving_log_dir():  
    dir = f"./results/solving_logs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_solving_log_path(name, K):
    dir = f"./results/solving_logs/{name}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return f"{dir}/{name}.{K}.log"

def get_aig_dir():
    if not os.path.exists("./data/aigs/"):
        os.makedirs("./data/aigs/")
    return "./data/aigs/"

def get_aig_path(name, K):
    aig_dir = get_aig_dir()
    return f"{aig_dir}/{name}.K_{K}.aig"

def get_solving_times_dir():
    dir = f"./results/solving_times/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_solving_times_path(name):
    dir = get_solving_times_dir()
    return f"{dir}/{name}.json"

def get_pysr_results_dir():
    dir = f"./results/pysr_results/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_pysr_results_path(name):
    dir = get_pysr_results_dir()
    return f"{dir}/{name}.jl"