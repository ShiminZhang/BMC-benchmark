# use pysr to do regression analysis

import os
import json
import pickle
import hashlib
import pysr
from category import get_all_instance_names
from paths import get_solving_times_dir, get_pysr_results_dir,get_results_dir, get_pysr_results_path, get_pysr_summary_path, get_pysr_cache_path, get_pysr_summary_path, get_sympy_summary_path, get_conclusion_path
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from Experiments.llm_refit_curve import llm_analysis, plot_original_vs_llm_results,llm_conclude_expression,plot_original_vs_equation,load_original_data,load_regression_equation,load_original_data_with_k, safe_evaluate_equation
from paths import get_plot_path
from Experiments.sympy_analysis import extract_leading_term
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
    from paths import get_pysr_cache_dir
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

def filter_k_step_data(k_data, x_data, y_data):
    """Filter data to only include consecutive k values with step = 1.
    
    Args:
        k_data: Array of k values
        x_data: Array of x values (CNF sizes)
        y_data: Array of y values (solving times)
    
    Returns:
        Filtered (k_data, x_data, y_data) arrays containing only consecutive k values
    """
    # Sort by k values
    sorted_indices = np.argsort(k_data)
    k_sorted = k_data[sorted_indices]
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]
    
    # Find consecutive points where k step == 1
    filtered_indices = [0]  # Always include the first point
    
    for i in range(1, len(k_sorted)):
        k_diff = k_sorted[i] - k_sorted[i-1]
        if k_diff == 1:
            # Continue when k step is exactly 1
            filtered_indices.append(i)
        elif k_diff > 1:
            # Stop when k step exceeds 1
            break
        # Skip if k_diff < 1 (duplicate or out of order)
    
    k_filtered = k_sorted[filtered_indices]
    x_filtered = x_sorted[filtered_indices]
    y_filtered = y_sorted[filtered_indices]
    
    print(f"Filtered data: {len(filtered_indices)}/{len(k_data)} points (k from {k_filtered[0]} to {k_filtered[-1]})")
    
    return k_filtered, x_filtered, y_filtered

def run_pysr(name, use_cache=False, save_equation=True, config_type="auto", x_data=None, y_data=None):
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
    
    # Use provided data or load from file
    if x_data is None or y_data is None:
        x_data, y_data = load_original_data(name)
    
    # Convert to numpy arrays if needed
    if not isinstance(x_data, np.ndarray):
        x_data = np.array(x_data)
    if not isinstance(y_data, np.ndarray):
        y_data = np.array(y_data)
    
    x = x_data.reshape(-1, 1)  # Reshape to 2D array for sklearn compatibility
    y = y_data
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

def plot_original_vs_equation_with_data(instance_name, type_of_equation: str, equations: dict, x_data, y_data, label=""):
    """Plot comparison of data and equation using provided data"""
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot original data points
    plt.scatter(x_data, y_data, color='black', alpha=0.7, s=50, 
               label='Original Data', zorder=5)
    print(f"Plotting equations: {equations}")
    
    # Plot equations
    for key, equation in equations.items():
        print(f"Plotting equation {key}: {equation}")
        x_plot = np.linspace(x_data.min(), x_data.max(), 1000)
        y_equation = safe_evaluate_equation(equation, x_plot)
        
        # Create unique label for each equation
        equation_label = f"{key}: {equation}"
        plt.plot(x_plot, y_equation, linewidth=2, label=equation_label, alpha=0.8)
    
    type_of_equation_label = f"Growth type: {type_of_equation}"
    plt.xlabel('Number of clauses in CNF formula', fontsize=12)
    plt.ylabel('Solving Time (seconds)', fontsize=12)
    plt.title(f'Comparison: Original Data vs Fitted Equation\nInstance: {instance_name}\n{type_of_equation_label}', 
             fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    figure_path = get_plot_path(instance_name)
    figure_path = figure_path.replace(".png", f"_{label}.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {figure_path}")
    plt.close()
    
    return figure_path

def plot_dual_comparison_with_data(instance_name, type_of_equation: str, equations: dict, k_data, x_data, y_data, label=""):
    """Plot dual comparison using provided data"""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    print(f"Plotting dual comparison for {instance_name}")
    print(f"Equations: {equations}")
    
    # ========== LEFT SUBPLOT: CNF Size vs Solving Time ==========
    ax1.scatter(x_data, y_data, color='black', alpha=0.7, s=50, 
               label='Original Data', zorder=5)
    
    # Plot equations on left subplot
    for key, equation in equations.items():
        print(f"Plotting equation {key}: {equation}")
        x_plot = np.linspace(x_data.min(), x_data.max(), 1000)
        y_equation = safe_evaluate_equation(equation, x_plot)
        
        # Create unique label for each equation
        equation_label = f"{key}: {equation[:50]}..." if len(equation) > 50 else f"{key}: {equation}"
        ax1.plot(x_plot, y_equation, linewidth=2, label=equation_label, alpha=0.8)
    
    ax1.set_xlabel('Number of clauses in CNF formula', fontsize=12)
    ax1.set_ylabel('Solving Time (seconds)', fontsize=12)
    ax1.set_title(f'CNF Size vs Solving Time\nInstance: {instance_name}', fontsize=14)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========== RIGHT SUBPLOT: K Value vs Solving Time ==========
    ax2.scatter(k_data, y_data, color='blue', alpha=0.7, s=50, 
               label='Original Data', zorder=5)
    
    # For k plot, we show the data points
    ax2.set_xlabel('K value', fontsize=12)
    ax2.set_ylabel('Solving Time (seconds)', fontsize=12)
    ax2.set_title(f'K Value vs Solving Time\nInstance: {instance_name}', fontsize=14)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Save the plot
    figure_path = get_plot_path(instance_name)
    figure_path = figure_path.replace(".png", f"_dual_{label}.png")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dual plot saved to: {figure_path}")
    plt.close()
    
    return figure_path

def run_pysr_and_check_with_llm(name, use_cache=False, config_type="auto"):
    model = run_pysr(name, use_cache=use_cache, save_equation=True, config_type=config_type)
    print(f"Best equation: {model.sympy()}")
    llm_analysis(name, use_cache=use_cache)

def run_pysr_and_plot(name, use_cache=False, config_type="auto"):
    """Run PySR regression and plot the results without LLM analysis
    
    Only uses data points where k_step = 1 (consecutive k values).
    Stops when the next data point's k value exceeds the previous by more than 1.
    
    Args:
        name: Instance name to analyze
        use_cache: Whether to use cached PySR model
        config_type: PySR configuration type (default: "auto")
    
    Returns:
        Path to the saved plot file, or None if plotting failed
    """
    # Load data with k values and filter to only include k_step = 1 consecutive data
    print(f"Loading data for {name} and filtering for k_step = 1...")
    k_data, x_data, y_data = load_original_data_with_k(name)
    
    # Filter data to only include consecutive k values with step = 1
    k_filtered, x_filtered, y_filtered = filter_k_step_data(k_data, x_data, y_data)
    
    if len(k_filtered) < 2:
        print(f"Not enough consecutive k data points ({len(k_filtered)}) for {name}")
        return None
    
    # Load or train the model with filtered data
    if use_cache:
        print(f"Using cache for {name}")
        model = load_cached_model(name)
    else:
        model = None
    
    # If no cached model, train with filtered data
    if model is None:
        print(f"Training model with {len(k_filtered)} filtered data points...")
        model = run_pysr(name, use_cache=use_cache, save_equation=True, config_type=config_type, 
                        x_data=x_filtered, y_data=y_filtered)
    
    if model is None:
        print(f"No model found for {name}")
        return None
    
    # Get the sympy equation and leading term
    sympy_summary = get_sympy_summary(name)
    equation = sympy_summary["equation"]
    leading_term = sympy_summary["leading_term"]
    
    print(f"PySR equation: {equation}")
    print(f"Leading term: {leading_term}")
    
    equations = {
        "pysr_equation": equation,
        "leading_term": leading_term
    }
    
    # Create single plot with data points and equation line
    print("\n📊 Creating single plot with data points and fitted equation...")
    single_plot_path = plot_original_vs_equation_with_data(name, "fitted", equations, 
                                                           x_filtered, y_filtered, label="pysr_plot")
    
    return single_plot_path, None
    if single_plot_path:
        print(f"✅ Single plot saved successfully")
    else:
        print("❌ Single plot creation failed")
    
    # Create dual plot (CNF size + K value)
    print("\n📊 Creating dual comparison plot (CNF size + K value)...")
    dual_plot_path = plot_dual_comparison_with_data(name, "fitted", equations, 
                                                     k_filtered, x_filtered, y_filtered, label="pysr_plot")
    
    if dual_plot_path:
        print(f"✅ Dual plot saved successfully")
    else:
        print("❌ Dual plot creation failed")
    
    return single_plot_path, dual_plot_path

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
    parser.add_argument("--pysr_plot", action="store_true", help="Run PySR and plot without LLM analysis")
    parser.add_argument("--pysr_plot_manage", action="store_true", help="Submit PySR plot job to SLURM")

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
    elif args.pysr_plot_manage:
        # Build the command for sbatch
        activate_python = "source ../../general/bin/activate"
        interested_names = get_all_instance_names()
        interested_names = sorted(list(interested_names))
        for name in interested_names:
            # Build command with proper flags
            cmd_parts = [
                activate_python,
                "&&",
                "python -m src.scripts.Experiments.regression_analysis",
                "--pysr_plot",
                f"--instance_name {name}",
                f"--config {args.config}"
            ]
            
            # Only add --use_cache flag if it's True (action="store_true")
            if args.use_cache:
                cmd_parts.append("--use_cache")
            
            cmd = " ".join(cmd_parts)
            
            # Build sbatch command with full output path
            output_path = f"logs/pysr_plot_{name}.out"
            sbatch_cmd = f"sbatch --job-name=pysr_plot_{name} --output={output_path} --mem=16g --time=4:00:00 --wrap=\"{cmd}\""
            
            print(f"Submitting job: {sbatch_cmd}")
            os.system(sbatch_cmd)
    elif args.pysr_plot:
        print(f"Running PySR and plotting for {args.instance_name}")
        print(f"Using cache: {pysr_use_cache}")
        single_plot_path, dual_plot_path = run_pysr_and_plot(args.instance_name, pysr_use_cache, args.config)
        if single_plot_path:
            print(f"✅ Single plot saved to: {single_plot_path}")
        else:
            print("❌ Single plot creation failed")
        if dual_plot_path:
            print(f"✅ Dual plot saved to: {dual_plot_path}")
        else:
            print("❌ Dual plot creation failed")
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