# use pysr to do regression analysis

import os
import json
import pickle
import hashlib
import pysr
from ..category import get_all_instance_names
from ..paths import get_solving_times_dir, get_pysr_results_dir, get_pysr_results_path, get_pysr_summary_path, get_pysr_cache_path
import numpy as np
import argparse
from .llm_refit_curve import llm_analysis, plot_original_vs_llm_results

def load_data(name):
    # load data from the solving_times directory
    solving_times_dir = get_solving_times_dir()
    solving_times_path = os.path.join(solving_times_dir, f"{name}.json")
    data = json.load(open(solving_times_path, "r"))
    return data

def get_data_hash(data):
    """Generate a hash of the input data for cache validation"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def load_cached_model(name, data_hash=None):
    """Load a cached PySR model if it exists and data hasn't changed"""
    cache_path = get_pysr_cache_path(name)
    cache_info_path = cache_path.replace('.pkl', '_info.json')
    
    if os.path.exists(cache_path) and os.path.exists(cache_info_path):
        try:
            # Load cache info
            with open(cache_info_path, 'r') as f:
                cache_info = json.load(f)
            
            # Check if data hash matches (if provided)
            if data_hash and cache_info.get('data_hash') != data_hash:
                print(f"Data has changed for {name}, cache invalid")
                return None
            
            # Load the model
            with open(cache_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded cached model for {name}")
            return model
        except Exception as e:
            print(f"Error loading cached model for {name}: {e}")
            return None
    return None

def save_model_to_cache(model, name, data_hash=None):
    """Save a PySR model to cache with metadata"""
    cache_path = get_pysr_cache_path(name)
    cache_info_path = cache_path.replace('.pkl', '_info.json')
    
    try:
        # Save the model
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save cache metadata
        cache_info = {
            'name': name,
            'timestamp': json.dumps(os.path.getmtime(cache_path)),
            'data_hash': data_hash,
            'model_type': str(type(model).__name__)
        }
        
        with open(cache_info_path, 'w') as f:
            json.dump(cache_info, f, indent=2)
        
        print(f"Saved model to cache for {name}")
    except Exception as e:
        print(f"Error saving model to cache for {name}: {e}")

def clear_cache(name=None):
    """Clear cached models. If name is None, clear all cache."""
    from ..paths import get_pysr_cache_dir
    import glob
    
    cache_dir = get_pysr_cache_dir()
    
    if name:
        # Clear specific cache
        cache_path = get_pysr_cache_path(name)
        cache_info_path = cache_path.replace('.pkl', '_info.json')
        
        for path in [cache_path, cache_info_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed {path}")
    else:
        # Clear all cache
        cache_files = glob.glob(os.path.join(cache_dir, "*"))
        for file_path in cache_files:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")

def save_best_equation(model, name):
    """Save only the best equation to a clean JSON file"""
    try:
        # Use model.sympy() to get the clean equation string
        best_equation = str(model.sympy())
        
        # Clean the equation string (remove extra whitespace, etc.)
        best_equation = best_equation.strip()
        
        # Save to JSON file
        equation_path = get_pysr_summary_path(name)
        with open(equation_path, 'w') as f:
            json.dump(best_equation, f, indent=None, separators=(',', ':'))
        
        print(f"Saved best equation for {name}: {best_equation}")
        return equation_path
        
    except Exception as e:
        print(f"Error saving equation for {name}: {e}")
        return None

def run_pysr(name, use_cache=False, save_equation=True):
    # Load data first to get hash
    data = load_data(name)
    data_hash = get_data_hash(data) if use_cache else None
    
    # Check if cached model exists and is valid
    if use_cache:
        cached_model = load_cached_model(name, data_hash)
        if cached_model is not None:
            # Also save equation if requested
            if save_equation:
                save_best_equation(cached_model, name)
            return cached_model
    
    print(f"Training new model for {name}...")
    
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
    
    # Save model to cache
    if use_cache:
        save_model_to_cache(model, name, data_hash)
    
    # Save best equation
    if save_equation:
        save_best_equation(model, name)
    
    return model

def batch_save_equations(names=None, use_cache=True):
    """Batch process instances and save their best equations"""
    if names is None:
        names = get_all_instance_names()
    
    results = {}
    
    for i, name in enumerate(names):
        print(f"\n=== Processing {name} ({i+1}/{len(names)}) ===")
        
        try:
            # Run PySR with caching and automatic equation saving
            model = run_pysr(name, use_cache=use_cache, save_equation=True)
            best_eq = str(model.sympy())
            results[name] = best_eq
            print(f"✓ Best equation: {best_eq}")
            
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            results[name] = None
    
    return results

def run_pysr_and_check_with_llm(name, use_cache=False):
    model = run_pysr(name, use_cache=use_cache, save_equation=True)
    print(f"Best equation: {model.sympy()}")
    llm_analysis(name, use_cache=use_cache)

def main():
    parser = argparse.ArgumentParser(description="Run PySR and check with LLM")
    parser.add_argument("--instance_name", type=str, default="oc8051gm0caddr", help="Instance name to analyze")
    parser.add_argument("--plot", action="store_true", help="Create comparison plot")
    parser.add_argument("--use_cache", action="store_true", help="Use cache")
    args = parser.parse_args()

    run_pysr_and_check_with_llm(args.instance_name, args.use_cache)
    if args.plot:
        print("\n📊 Creating comparison plot...")
        plot_path = plot_original_vs_llm_results(args.instance_name)
        if plot_path:
            print(f"✅ Plot saved to: {plot_path}")
        else:
            print("❌ Plot creation failed")

    # names = get_all_instance_names()

    # for name in names:
    #     print(f"\n=== Processing {name} ===")
        
    #     # Run PySR with caching and automatic equation saving
    #     model = run_pysr(name, use_cache=True, save_equation=True)
        
    #     print(f"Best equation: {model.sympy()}")

if __name__ == "__main__":
    main()