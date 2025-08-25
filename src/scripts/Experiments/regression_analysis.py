# use pysr to do regression analysis

import os
import json
import pysr
from ..category import get_all_instance_names
from ..paths import get_solving_times_dir, get_pysr_results_dir, get_pysr_results_path
import numpy as np

def load_data(name):
    # load data from the solving_times directory
    solving_times_dir = get_solving_times_dir()
    solving_times_path = os.path.join(solving_times_dir, f"{name}.json")
    data = json.load(open(solving_times_path, "r"))
    return data

def run_pysr(name):
    data = load_data(name)
    # Convert data to proper format for PySR
    # data.items() gives us (key, value) pairs, we need to separate them
    items = list(data.items())
    # Convert string keys to numeric values
    x = [float(item[0]) for item in items]  # Convert string keys to float
    y = [item[1] for item in items]  # Extract values as targets
    
    # Convert to numpy arrays if needed
    x = np.array(x).reshape(-1, 1)  # Reshape to 2D array for sklearn compatibility
    y = np.array(y)
    
    output_dir = get_pysr_results_dir() 
    os.makedirs(output_dir, exist_ok=True)
    
    # use pysr to do regression analysis
    model = pysr.PySRRegressor(
        niterations=100,
        output_directory=get_pysr_results_path(name),
        tempdir=os.path.join(output_dir, "temp")
    )
    model.fit(x, y)
    return model

def main():
    names = get_all_instance_names()
    for name in names:
        print(name)
        model = run_pysr(name)
        print(model)
        exit()

if __name__ == "__main__":
    main()