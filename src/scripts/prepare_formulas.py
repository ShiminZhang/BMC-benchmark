import os
import multiprocessing
from GenericRA import LOG, LOG_TAG, TOGGLE_SHOWLOG, REG_TAG

def run_formula(name, K, solver, limit):
    LOG(f"Running formula {name} with K={K} up to limit {limit}")
    pass

def generate_cnf(name, K):
    LOG(f"Generating formula {name} with K={K}")
    
    pass

def generate_and_run_up_to_limit(name, K, solver, limit):
    LOG(f"Generating and running {name} with K={K} up to limit {limit}")
    for i in range(1, limit + 1):
        generate_cnf(name, i)
        run_formula(name, i, solver, limit)

def main():
    TOGGLE_SHOWLOG(True)
    processes = []
    names = ["6s0", "6s4"]
    time_limit = 1600
    solver = "./solvers/bin/cadical"
    for name in names:
        process = multiprocessing.Process(target=generate_and_run_up_to_limit, args=(name, 1, solver, time_limit))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()