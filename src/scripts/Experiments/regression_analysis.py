# use pysr to do regression analysis

import os
import json
import pickle
import hashlib
import pysr
from ..category import get_all_instance_names
from ..paths import get_solving_times_dir, get_pysr_results_dir,get_results_dir, get_pysr_results_path, get_pysr_summary_path, get_pysr_cache_path, get_pysr_summary_path, get_sympy_summary_path, get_conclusion_path
import numpy as np
import argparse
from .llm_refit_curve import llm_analysis, plot_original_vs_llm_results,llm_conclude_expression,plot_original_vs_equation,load_original_data,load_regression_equation
from .sympy_analysis import extract_leading_term
import multiprocessing
from multiprocessing import Pool, cpu_count
import time

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
    # cache_info_path = cache_path.replace('.pkl', '_info.json')
    
    # if os.path.exists(cache_path) and os.path.exists(cache_info_path):
    if os.path.exists(cache_path):
        try:
            # # Load cache info
            # with open(cache_info_path, 'r') as f:
            #     cache_info = json.load(f)
            
            # # Check if data hash matches (if provided)
            # if data_hash and cache_info.get('data_hash') != data_hash:
            #     print(f"Data has changed for {name}, cache invalid")
            #     return None
            
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

def get_pysr_config(config_type="balanced"):
    """
    Get PySR configuration based on the specified type.
    
    Args:
        config_type (str): Configuration type
            - "auto": Let PySR automatically discover the best mathematical form (recommended)
            - "balanced": Good balance between accuracy and simplicity, lets PySR discover the best form
            - "exponential": Optimized for exponential/logarithmic functions
            - "simple": Very simple equations, minimal overfitting
            - "accurate": High accuracy, may be more complex
            - "custom": Custom configuration for experimentation
    
    Returns:
        dict: Configuration parameters for PySR
    """
    
    base_config = {
        # "niterations": 200,
        # "population_size": 25,
        # "ncycles_per_iteration": 50,
        # "warm_start": True,
        # "verbosity": 1,
        # "timeout_in_seconds": 300,
        # "select_k_features": 1,
    }
    
    if config_type == "auto":
        # Configuration that lets PySR automatically discover the best form
        return {
            **base_config,
            "niterations": 250,  # More iterations for better exploration
            "binary_operators": ["+", "-", "*", "^"],  # All basic operations
            "unary_operators": ["exp"],  # Include exponential functions
            "constraints": {"maxdepth": 5},
            # "maxsize": 20,  # Allow enough complexity for exponential terms if needed
            # "maxdepth": 10,
            # "parsimony": 0.06,  # Lower parsimony to allow exponential terms if beneficial
            # "alpha": 0.06,  # Lower regularization to allow more complex forms
            # "elementwise_loss": "loss(x, y) = abs(x - y)",
            # "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-8 && complexity < 15",
            # # Intelligent search parameters
            "warm_start": True,  # Use previous solutions as starting point
            # "populations": 4,  # Multiple populations for better exploration
            # "fraction_replaced": 0.08,  # Conservative replacement to maintain good solutions
            # "tournament_selection_n": 5,  # Tournament selection for better diversity
        }
    
    elif config_type == "balanced":
        return {
            **base_config,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log", "sqrt", "abs"],  # Include exponential functions but don't force them
            "maxsize": 18,  # Allow enough complexity for exponential terms if needed
            "maxdepth": 9,
            "parsimony": 0.08,  # Balanced parsimony - not too high, not too low
            "alpha": 0.08,  # Balanced regularization
            "elementwise_loss": "loss(x, y) = abs(x - y)",
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-7 && complexity < 12",
            # Let PySR naturally discover if exponential terms are needed
            "warm_start": True,  # Use previous solutions as starting point
            "npopulations": 3,  # Multiple populations for better exploration
            "fraction_replaced": 0.1,  # Conservative replacement to maintain good solutions
        }
    
    elif config_type == "exponential":
        return {
            **base_config,
            "niterations": 300,  # More iterations for complex functions
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log", "sqrt", "abs", "sin", "cos"],  # Include trig functions
            "maxsize": 20,  # Allow slightly larger equations for exponential functions
            "maxdepth": 10,
            "parsimony": 0.05,  # Lower parsimony to allow more complex exponential forms
            "alpha": 0.05,  # Lower regularization for exponential functions
            "elementwise_loss": "loss(x, y) = abs(x - y)",
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-8 && complexity < 15",
        }
    
    elif config_type == "simple":
        return {
            **base_config,
            "niterations": 150,
            "binary_operators": ["+", "-", "*"],  # No division to avoid complexity
            "unary_operators": ["abs"],  # Only absolute value
            "maxsize": 8,  # Very small equations
            "maxdepth": 4,
            "parsimony": 0.3,  # High parsimony for simplicity
            "alpha": 0.2,  # High regularization
            "elementwise_loss": "loss(x, y) = abs(x - y)",
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-5 && complexity < 5",
        }
    
    elif config_type == "accurate":
        return {
            **base_config,
            "niterations": 400,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log", "sqrt", "abs", "sin", "cos", "tan"],
            "maxsize": 25,
            "maxdepth": 12,
            "parsimony": 0.02,  # Very low parsimony for accuracy
            "alpha": 0.01,  # Low regularization
            "elementwise_loss": "loss(x, y) = (x - y)^2",  # MSE loss for accuracy
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-10",
        }
    
    elif config_type == "custom":
        # Custom configuration for experimentation
        return {
            **base_config,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log", "sqrt", "abs"],
            "maxsize": 18,
            "maxdepth": 9,
            "parsimony": 0.08,
            "alpha": 0.08,
            "elementwise_loss": "loss(x, y) = abs(x - y)",
            "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-7 && complexity < 12",
        }
    
    else:
        raise ValueError(f"Unknown config_type: {config_type}. Use 'auto', 'balanced', 'exponential', 'simple', 'accurate', or 'custom'")

def run_pysr(name, use_cache=False, save_equation=True, config_type="auto"):
    # Load data first to get hash
    
    # Check if cached model exists and is valid
    if use_cache:
        print(f"Using cache for {name} with save_equation: {save_equation}")

    data = load_data(name)
    if len(data) == 0:
        print(f"No data found for {name}")
        return None
    elif len(data) <= 5:
        print(f"Not enough data ({len(data)}) found for {name}, using default configuration")
        return None

    data_hash = get_data_hash(data) if use_cache else None
    if use_cache:
        print(f"Loading cached model for {name} with data hash: {data_hash}")
        cached_model = load_cached_model(name, data_hash)
        if cached_model is not None:
            # Also save equation if requested
            if save_equation:
                save_best_equation(cached_model, name)
            return cached_model
    
    print(f"Training new model for {name} with {config_type} configuration...")
    
    # Convert data to proper format for PySR
    # data.items() gives us (key, value) pairs, we need to separate them
    items = list(data.items())
    # Convert string keys to numeric values
    
    # k as x
    k = [float(item[0]) for item in items]  # Convert string keys to float
    x_data, y_data = load_original_data(name)
    # # size as x
    # size = [float(item[1]["size_of_cnf"]) for item in items]  # Convert string keys to float

    # # time as y
    # time = [float(item[1]["solving_time"]) for item in items]  # Extract values as targets
    
    # x = size
    # y = time  # Extract values as targets
    
    # # Convert to numpy arrays if needed
    x = np.array(x_data).reshape(-1, 1)  # Reshape to 2D array for sklearn compatibility
    y = np.array(y_data)
    # print(f"x: {x}")
    # print(f"y: {y}")
    output_dir = get_pysr_results_dir() 
    os.makedirs(output_dir, exist_ok=True)
    
    # Get configuration based on type
    config = get_pysr_config(config_type)
    
    # Create PySR model with the selected configuration
    model = pysr.PySRRegressor(
        output_directory=get_pysr_results_path(name),
        tempdir=os.path.join(output_dir, "temp"),
        **config
    )
    
    # Fit the model
    model.fit(x, y)
    
    # Save model to cache
    if use_cache:
        save_model_to_cache(model, name, data_hash)
    
    # Save best equation
    if save_equation:
        save_best_equation(model, name)
    term = extract_leading_term(model.sympy())
    result = {}
    result["equation"] = str(model.sympy())
    result["leading_term"] = str(term)
    print(f"Leading term: {term}")
    print(f"Saved to: {get_sympy_summary_path(name)}")
    with open(get_sympy_summary_path(name), "w") as f:
        json.dump(result, f, indent=4)
    return model

def get_sympy_summary(name):
    path = get_sympy_summary_path(name)
    with open(path, "r") as f:
        data = json.load(f)
    return data

def batch_save_equations(names=None, use_cache=True, config_type="auto"):
    """Batch process instances and save their best equations"""
    if names is None:
        names = get_all_instance_names()
    
    results = {}
    
    for i, name in enumerate(names):
        print(f"\n=== Processing {name} ({i+1}/{len(names)}) with {config_type} config ===")
        
        try:
            # Run PySR with caching and automatic equation saving
            model = run_pysr(name, use_cache=use_cache, save_equation=True, config_type=config_type)
            best_eq = str(model.sympy())
            results[name] = best_eq
            print(f"✓ Best equation: {best_eq}")
            
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            results[name] = None
    
    return results

def run_pysr_and_check_with_llm(name, use_cache=False, config_type="auto"):
    model = run_pysr(name, use_cache=use_cache, save_equation=True, config_type=config_type)
    print(f"Best equation: {model.sympy()}")
    llm_analysis(name, use_cache=use_cache)

def run_pysr_and_conclude_with_llm(name, use_cache=False, config_type="auto", plot=False):
    if use_cache:
        print(f"Using cache for {name}")
        model = load_cached_model(name)
    model = model or run_pysr(name, use_cache=use_cache, save_equation=True, config_type=config_type)

    if model is None:
        print(f"No model found for {name}")
        return
    sympy_summary = get_sympy_summary(name)
    equation = sympy_summary["equation"]
    leading_term = sympy_summary["leading_term"]
    conclusion = llm_conclude_expression(name, equation, leading_term, use_cache)
    if conclusion is None:
        print(f"No conclusion found for {name}")
        return
    if plot:
        print("\n📊 Creating comparison plot...")
        llm_equation = conclusion["llm_upper_bound"]
        equations ={"pysr_equation": equation, "leading_term": leading_term, "upper_bound": llm_equation}
        plot_original_vs_equation(name, conclusion["type_of_equation"], equations, label="LLM_included")
        equations ={"pysr_equation": equation}
        plot_original_vs_equation(name, conclusion["type_of_equation"], equations, label="original_only")

def process_single_instance(args):
    """Process a single instance by calling run_pysr_and_conclude_with_llm"""
    instance_name, use_cache, config_type, plot = args
    
    # Capture stdout and stderr
    import sys
    from io import StringIO
    import contextlib
    
    # Create string buffers to capture output
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        print(f"🔄 Processing {instance_name}...")
        
        # Capture stdout and stderr from the function call
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            run_pysr_and_conclude_with_llm(instance_name, use_cache, config_type, plot)
        
        # Get the captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        print(f"✅ {instance_name}: Completed successfully")
        
        # Print the captured output with instance prefix
        if stdout_output.strip():
            print(f"\n--- STDOUT for {instance_name} ---")
            for line in stdout_output.strip().split('\n'):
                print(f"[{instance_name}] {line}")
            print(f"--- END STDOUT for {instance_name} ---\n")
        
        if stderr_output.strip():
            print(f"\n--- STDERR for {instance_name} ---")
            for line in stderr_output.strip().split('\n'):
                print(f"[{instance_name}] {line}")
            print(f"--- END STDERR for {instance_name} ---\n")
        
        return {
            "instance_name": instance_name, 
            "status": "success",
            "stdout": stdout_output,
            "stderr": stderr_output
        }
    except Exception as e:
        print(f"❌ {instance_name}: {e}")
        return {
            "instance_name": instance_name, 
            "status": "error", 
            "error": str(e),
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue()
        }

def parallel_llm_conclude_expression(instance_names=None, use_cache=False, config_type="auto", plot=False, max_workers=None):
    """Run LLM conclude expression analysis on multiple instances in parallel
    
    Args:
        instance_names: List of instance names to process. If None, processes all instances.
        use_cache: Whether to use cached results
        config_type: PySR configuration type
        plot: Whether to create comparison plots
        max_workers: Maximum number of parallel workers. If None, uses CPU count.
        
    Returns:
        Dict containing results for all processed instances
    """
    if instance_names is None:
        print("📋 Getting all instance names...")
        instance_names = list(get_all_instance_names())
    
    if not instance_names:
        print("⚠️  No instances found to process")
        return {}
    
    print(f"🚀 Starting parallel LLM conclude expression analysis")
    print(f"   Instances: {len(instance_names)}")
    print(f"   Use cache: {use_cache}")
    print(f"   Create plots: {plot}")
    print(f"   Max workers: {max_workers or cpu_count()}")
    print("=" * 60)
    
    # Prepare arguments for each instance
    instance_args = [(name, use_cache, config_type, plot) for name in instance_names]
    
    # Set up multiprocessing
    if max_workers is None:
        max_workers = min(cpu_count(), len(instance_names))
    
    results = {}
    start_time = time.time()
    
    try:
        # Process all instances in parallel
        with Pool(processes=max_workers) as pool:
            all_results = pool.map(process_single_instance, instance_args)
        
        # Convert to dictionary
        for result in all_results:
            results[result["instance_name"]] = result
    
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        return {"error": str(e)}
    
    # Calculate statistics
    end_time = time.time()
    duration = end_time - start_time
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PARALLEL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {duration:.2f} seconds")
    print(f"📈 Total instances: {len(instance_names)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    
    if success_count > 0:
        avg_time = duration / success_count
        print(f"⚡ Average time per successful instance: {avg_time:.2f} seconds")
    
    # Show instances with output
    instances_with_stdout = [name for name, result in results.items() if result.get("stdout", "").strip()]
    instances_with_stderr = [name for name, result in results.items() if result.get("stderr", "").strip()]
    
    if instances_with_stdout:
        print(f"📝 Instances with stdout output: {len(instances_with_stdout)}")
    if instances_with_stderr:
        print(f"⚠️  Instances with stderr output: {len(instances_with_stderr)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run PySR and check with LLM")
    parser.add_argument("--instance_name", type=str, default="oc8051gm0caddr", help="Instance name to analyze")
    parser.add_argument("--plot", action="store_true", help="Create comparison plot")
    parser.add_argument("--use_cache", action="store_true", help="Use cache")
    parser.add_argument("--use_pysr_cache", action="store_true", help="Use PySR cache")
    parser.add_argument("--regression_only", action="store_true", help="Only run regression analysis")
    parser.add_argument("--config", type=str, default="auto", 
                       choices=["auto", "balanced", "exponential", "simple", "accurate", "custom"],
                       help="PySR configuration type (default: auto - lets PySR discover the best form)")
    parser.add_argument("--llm_analysis", action="store_true", help="Only run LLM analysis")
    parser.add_argument("--find_insufficient_data", action="store_true", help="Find insufficient data")
    parser.add_argument("--sequential_conclude", action="store_true", help="Sequential LLM conclude expression")
    parser.add_argument("--llm_conclude_expression", action="store_true", help="Only run LLM conclude expression")
    parser.add_argument("--parallel_llm_conclude_expression", action="store_true", help="Parallel LLM conclude expression")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers for parallel processing")

    args = parser.parse_args()

    pysr_use_cache = args.use_pysr_cache or args.use_cache

    if args.parallel_llm_conclude_expression:
        interested_names = get_all_instance_names()
        interested_names = sorted(list(interested_names))
        # interested_names = ["6s0", "6s109", "bob2"]
        print("🚀 Starting parallel LLM conclude expression analysis...")
        results = parallel_llm_conclude_expression(
            instance_names=interested_names,
            use_cache=args.use_cache,
            config_type=args.config,
            plot=args.plot,
            max_workers=args.max_workers
        )
        #save the results to temp file
        results_dir = get_results_dir()
        output_file = os.path.join(results_dir, "parallel_llm_conclude_expression.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        # Save stdout/stderr to separate files for easier debugging
        stdout_dir = os.path.join(results_dir, "parallel_stdout")
        os.makedirs(stdout_dir, exist_ok=True)
        
        for instance_name, result in results.items():
            if result.get("stdout", "").strip():
                stdout_file = os.path.join(stdout_dir, f"{instance_name}_stdout.txt")
                with open(stdout_file, "w") as f:
                    f.write(result["stdout"])
            
            if result.get("stderr", "").strip():
                stderr_file = os.path.join(stdout_dir, f"{instance_name}_stderr.txt")
                with open(stderr_file, "w") as f:
                    f.write(result["stderr"])
        
        print(f"✅ Parallel processing completed. Processed {len(results)} instances.")
        print(f"📁 Results saved to: {output_file}")
        print(f"📁 Individual stdout/stderr saved to: {stdout_dir}")
        return
        
    if args.find_insufficient_data:
        results_dir = get_results_dir()
        output_file = os.path.join(results_dir, "insufficient_data.txt")
        results = []
        for name in get_all_instance_names():
            if f"{name}.json" not in os.listdir(get_solving_times_dir()):
                continue
            data = load_data(name)
            if len(data) <= 5:
                results.append(name)
        with open(output_file, "w") as f:
            f.write(str(results))
        print(f"Insufficient data for {len(results)} instances saved to {output_file}")
        return

    if args.regression_only:
        model = run_pysr(args.instance_name, use_cache=pysr_use_cache, save_equation=True, config_type=args.config)
        print(f"Best equation: {model.sympy()}")
        print(model)
    elif args.llm_analysis:
        run_pysr_and_check_with_llm(args.instance_name, args.use_cache, args.config)
        if args.plot:
            print("\n📊 Creating comparison plot...")
            plot_path = plot_original_vs_llm_results(args.instance_name)
            if plot_path:
                print(f"✅ Plot saved to: {plot_path}")
            else:
                print("❌ Plot creation failed")
    elif args.sequential_conclude:
        interested_names = get_all_instance_names()
        interested_names = sorted(list(interested_names))
        for name in interested_names:
            if f"{name}.json" not in os.listdir(get_solving_times_dir()):
                continue
            equation = load_regression_equation(name)
            if equation is None:
                if not run_pysr(name, use_cache=args.use_cache, save_equation=True, config_type=args.config):
                    continue
            sympy_summary = get_sympy_summary(name)
            # equation = sympy_summary["equation"]
            leading_term = sympy_summary["leading_term"]
            # conclusion = llm_conclude_expression(name, equation, leading_term, True)
            conclusion_path = get_conclusion_path(name)
            if os.path.exists(conclusion_path):
                with open(conclusion_path, "r") as f:
                    conclusion = json.load(f)
                if conclusion["original_equation"] != equation:
                    print(f"Original equation: {equation}")
                    print(f"LLM concluded equation: {conclusion['original_equation']}")
                    conclusion["original_equation"] = equation
                    with open(get_conclusion_path(name), "w") as f:
                        json.dump(conclusion, f, indent=4)
            else:
                print(f"Conclusion not found for {name}, creating new one")
                conclusion = {}
                conclusion["original_equation"] = equation
                conclusion["leading_term"] = leading_term
                conclusion["type_of_equation"] = "NA"
                conclusion["llm_upper_bound"] = "NA"
                conclusion["llm_complexity"] = "NA"
                conclusion["llm_confidence"] = 1.0
                with open(get_conclusion_path(name), "w") as f:
                    json.dump(conclusion, f, indent=4)
                print(f"Conclusion saved to {get_conclusion_path(name)}")        
                equations ={"pysr_equation": equation}
                plot_original_vs_equation(name, conclusion["type_of_equation"], equations, label="original_only")

    elif args.llm_conclude_expression:
        print(f"Using cache: {args.use_cache}")
        run_pysr_and_conclude_with_llm(args.instance_name, args.use_cache, args.config, args.plot) 
    else:
        names = get_all_instance_names()
        for name in names:
            if f"{name}.json" not in os.listdir(get_solving_times_dir()):
                continue
            
            if os.path.exists(get_sympy_summary_path(name)) and False:
                print(f"Loading leading term from {get_sympy_summary_path(name)}")
                with open(get_sympy_summary_path(name), "r") as f:
                    term = json.load(f)["leading_term"]
                print(f"Leading term: {term}")
            else:
                model = run_pysr(name, use_cache=args.use_cache, save_equation=True, config_type=args.config)
                print(f"Model: {model.sympy()}")
                term = extract_leading_term(model.sympy())
                result = {}
                result["equation"] = str(model.sympy())
                result["leading_term"] = str(term)
                print(f"Leading term: {term}")
                print(f"Saved to: {get_sympy_summary_path(name)}")
                with open(get_sympy_summary_path(name), "w") as f:
                    json.dump(result, f, indent=4)

                

if __name__ == "__main__":
    main()