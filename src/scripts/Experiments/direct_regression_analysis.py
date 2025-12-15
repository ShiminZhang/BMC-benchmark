"""
Direct Regression Analysis Script

This script performs regression analysis on solving time data by fitting different models
(linear, polynomial, exponential) and comparing their performance using R-squared metrics.
"""

import os
import json
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Any, Optional
from regression_analysis import load_original_data_with_k,filter_k_step_data
from paths import get_solving_times_path, get_plots_dir
from category import get_all_instance_names


def make_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON dumping.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_solving_data(instance_name: str) -> Tuple[np.ndarray, np.ndarray]:
    k_data, x_data, y_data = load_original_data_with_k(instance_name)
    k_filtered, x_filtered, y_filtered = filter_k_step_data(k_data, x_data, y_data)
    # 运行最大，保证非降
    y_running_max = np.maximum.accumulate(y_filtered)

    # 找到 running max 发生上升的拐点索引（含首尾）
    n = len(y_running_max)
    change_idxs = np.r_[0, np.flatnonzero(np.diff(y_running_max) != 0) + 1, n - 1]
    change_idxs = np.unique(change_idxs)

    # 以 x 作为插值自变量，分段线性插值连接相邻拐点
    x_knots = x_filtered[change_idxs]
    y_knots = y_running_max[change_idxs].astype(float)


    # 最后一段插值到 1600（若已超过则不降低）
    if len(x_data) < 100:
        if y_knots.size > 0 and y_knots[-1] < 1600.0:
            y_knots[-1] = 1600.0

    # 去除可能的重复 x 结点，避免 np.interp 报错
    x_knots_unique, unique_idx = np.unique(x_knots, return_index=True)
    y_knots_unique = y_knots[unique_idx]

    y_interpolated = np.interp(x_filtered, x_knots_unique, y_knots_unique)

    return x_filtered, y_interpolated
    # keep only points where y strictly exceeds previous max
    x_arr = np.array(x_filtered)
    y_arr = np.array(y_filtered)
    keep_mask = []
    prev_max = -np.inf
    for val in y_arr:
        if val > prev_max:
            keep_mask.append(True)
            prev_max = val
        else:
            keep_mask.append(False)
    keep_mask = np.array(keep_mask, dtype=bool)
    return x_arr[keep_mask], y_arr[keep_mask]


def exponential_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential function for curve fitting.
    
    Args:
        x: Input values
        a: Amplitude parameter
        b: Growth rate parameter
        
    Returns:
        Exponential function values
    """
    return a * np.exp(b * x)


def exponential_func_with_offset(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential function with offset term for curve fitting: y = a * exp(b * x) + c
    """
    return a * np.exp(b * x) + c


def fit_models(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
    """
    Fit different regression models to the data.
    
    Args:
        x_data: Input features (CNF size)
        y_data: Target values (solving time)
        
    Returns:
        Dictionary containing fitted models and their predictions
    """
    results = {}
    
    # Check if we have enough data points for reliable fitting
    if len(x_data) < 3:
        print(f"Warning: Only {len(x_data)} data points available. Some models may not fit reliably.")
        # For very few points, only try linear model
        try:
            linear_coeffs = np.polyfit(x_data, y_data, 1)
            y_pred_linear = np.poly1d(linear_coeffs)(x_data)
            r2_linear = r2_score(y_data, y_pred_linear)
            results['linear'] = {
                'coefficients': linear_coeffs,
                'predictions': y_pred_linear,
                'r2_score': r2_linear
            }
        except Exception as e:
            print(f"Warning: Linear model fitting failed: {e}")
            results['linear'] = {'r2_score': -np.inf}
        
        # Skip other models for insufficient data
        results['polynomial'] = {'r2_score': -np.inf}
        results['exponential'] = {'r2_score': -np.inf}
        return results
    
    # Linear model
    try:
        linear_coeffs = np.polyfit(x_data, y_data, 1)
        y_pred_linear = np.poly1d(linear_coeffs)(x_data)
        r2_linear = r2_score(y_data, y_pred_linear)
        results['linear'] = {
            'coefficients': linear_coeffs,
            'predictions': y_pred_linear,
            'r2_score': r2_linear
        }
    except Exception as e:
        print(f"Warning: Linear model fitting failed: {e}")
        results['linear'] = {'r2_score': -np.inf}
    
    # Polynomial model (adjust degree based on data size)
    try:
        # Use lower degree for fewer data points to avoid overfitting
        poly_degree = 3
        if poly_degree < 1:
            poly_degree = 1
            
        poly_coeffs = np.polyfit(x_data, y_data, poly_degree)
        y_pred_poly = np.poly1d(poly_coeffs)(x_data)
        r2_poly = r2_score(y_data, y_pred_poly)
        results['polynomial'] = {
            'coefficients': poly_coeffs,
            'predictions': y_pred_poly,
            'r2_score': r2_poly,
            'degree': poly_degree
        }
    except Exception as e:
        print(f"Warning: Polynomial model fitting failed: {e}")
        results['polynomial'] = {'r2_score': -np.inf}
    
    # Exponential model
    try:
        # Normalize x only to improve numerical stability; keep y in original scale
        x_min, x_max = np.min(x_data), np.max(x_data)
        x_range = x_max - x_min + 1e-10
        x_norm = (x_data - x_min) / x_range

        y_min, y_max = np.min(y_data), np.max(y_data)
        y_range = y_max - y_min + 1e-10

        # Initial guesses for parameters of y = a * exp(b * x_norm) + c
        # Assume baseline near min(y), amplitude near (max - min)
        c0 = float(y_min)
        a0 = float(max(y_max - c0, 1e-6))
        b0 = 1.0  # moderate growth in x_norm ∈ [0,1]
        initial_guess = [a0, b0, c0]

        # Bounds: a >= 0, b in a reasonable range for x_norm, c within a loose band around observed y
        lower_bounds = [0.0, -50.0, y_min - 2.0 * y_range]
        upper_bounds = [1e12, 50.0, y_max + 2.0 * y_range]

        popt_exp, pcov_exp = curve_fit(
            exponential_func_with_offset,
            x_norm,
            y_data,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )

        y_pred_exp = exponential_func_with_offset(x_norm, *popt_exp)

        # Check for invalid values (inf, nan)
        if np.any(np.isinf(y_pred_exp)) or np.any(np.isnan(y_pred_exp)):
            raise ValueError("Exponential model produced invalid predictions")

        r2_exp = r2_score(y_data, y_pred_exp)
        results['exponential'] = {
            'parameters': popt_exp,
            'covariance': pcov_exp,
            'predictions': y_pred_exp,
            'r2_score': r2_exp
        }
        print(f"Exponential model fitted successfully with R² = {r2_exp:.4f}")
        
    except (RuntimeError, ValueError, OverflowError) as e:
        print(f"Warning: Exponential model fitting failed: {e}")
        # Try alternative approach with log transformation
        try:
            print("Trying alternative exponential fitting with log transformation...")
            # Shift y by baseline to be positive, fit log(y_shift) ~ b * x_norm + log(a)
            x_min, x_max = np.min(x_data), np.max(x_data)
            x_range = x_max - x_min + 1e-10
            x_norm = (x_data - x_min) / x_range

            baseline = float(np.min(y_data))
            y_shift = y_data - baseline + 1e-10
            if np.all(y_shift > 0):
                log_y = np.log(y_shift)
                linear_coeffs = np.polyfit(x_norm, log_y, 1)
                a = float(np.exp(linear_coeffs[1]))
                b = float(linear_coeffs[0])
                c = float(baseline)
                y_pred_exp = a * np.exp(b * x_norm) + c
                r2_exp = r2_score(y_data, y_pred_exp)

                results['exponential'] = {
                    'parameters': [a, b, c],
                    'covariance': None,
                    'predictions': y_pred_exp,
                    'r2_score': r2_exp
                }
                print(f"Alternative exponential model (with offset) fitted with R² = {r2_exp:.4f}")
            else:
                raise ValueError("Cannot apply log transformation due to non-positive shifted values")
        except Exception as e2:
            print(f"Alternative exponential fitting also failed: {e2}")
            results['exponential'] = {'r2_score': -np.inf}
    
    return results


def find_best_model(results: Dict[str, Any]) -> Tuple[str, float]:
    """
    Find the best performing model based on R-squared score.
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        Tuple of (best_model_name, best_r2_score)
    """
    valid_models = {k: v['r2_score'] for k, v in results.items()
                    if isinstance(v, dict) and 'r2_score' in v and v['r2_score'] is not None and not np.isnan(v['r2_score'])}
    
    if not valid_models:
        return "None", -np.inf
    
    # Apply selection bonus: linear gets +0.1 for tie-breaking preference
    selection_scores: Dict[str, float] = {}
    for model_name, r2 in valid_models.items():
        try:
            bonus = 0.1 if model_name == 'linear' else 0.0
            selection_scores[model_name] = float(r2) + bonus
        except Exception:
            continue
    
    if not selection_scores:
        return "None", -np.inf
    
    best_model = max(selection_scores, key=selection_scores.get)
    # Return the raw R^2 for the best model (without bonus)
    raw_best_r2 = float(results[best_model]['r2_score']) if best_model in results and 'r2_score' in results[best_model] else float(valid_models.get(best_model, -np.inf))
    return best_model, raw_best_r2


def plot_results(x_data: np.ndarray, y_data: np.ndarray, results: Dict[str, Any], 
                instance_name: str, save_plot: bool = True) -> None:
    """
    Create and optionally save a plot comparing different models.
    
    Args:
        x_data: Input features
        y_data: Target values
        results: Dictionary containing model results
        instance_name: Name of the instance for plot title
        save_plot: Whether to save the plot to file
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(x_data, y_data, label='Original Data', color='blue', alpha=0.6, s=30)
    
    # Plot linear model
    if 'linear' in results and 'predictions' in results['linear']:
        plt.plot(x_data, results['linear']['predictions'], 
                label=f'Linear (R²={results["linear"]["r2_score"]:.4f})', 
                color='green', linestyle='--', linewidth=2)
    
    # Plot polynomial model
    if 'polynomial' in results and 'predictions' in results['polynomial']:
        plt.plot(x_data, results['polynomial']['predictions'], 
                label=f'Polynomial (R²={results["polynomial"]["r2_score"]:.4f})', 
                color='red', linestyle='-', linewidth=2)
    
    # Plot exponential model
    if 'exponential' in results and 'predictions' in results['exponential']:
        plt.plot(x_data, results['exponential']['predictions'], 
                label=f'Exponential (R²={results["exponential"]["r2_score"]:.4f})', 
                color='purple', linestyle='-.', linewidth=2)
    
    plt.title(f'Regression Analysis for {instance_name}', fontsize=16)
    plt.xlabel('CNF Size', fontsize=12)
    plt.ylabel('Solving Time (seconds)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_plot:
        plots_dir = get_plots_dir()
        plot_path = os.path.join(plots_dir, f"{instance_name}_regression_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    # plt.show()


def print_results(results: Dict[str, Any], best_model: str, best_r2: float) -> None:
    """
    Print the analysis results in a formatted way.
    
    Args:
        results: Dictionary containing model results
        best_model: Name of the best model
        best_r2: R-squared score of the best model
    """
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS RESULTS")
    print("="*60)
    
    print("\nModel R-squared Scores:")
    print("-" * 30)
    
    for model_name, model_data in results.items():
        if 'r2_score' in model_data and not np.isnan(model_data['r2_score']):
            print(f"{model_name.capitalize():<12}: {model_data['r2_score']:.4f}")
        else:
            print(f"{model_name.capitalize():<12}: Failed to fit")
    
    print(f"\nBest Model: {best_model.capitalize()}")
    print(f"Best R² Score: {best_r2:.4f}")
    print("="*60)


def analyze_instance(instance_name: str, save_plot: bool = True) -> Dict[str, Any]:
    """
    Perform complete regression analysis for a single instance.
    
    Args:
        instance_name: Name of the instance to analyze
        save_plot: Whether to save the plot
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Analyzing instance: {instance_name}")
    
    try:
        # Load data
        x_data, y_data = load_solving_data(instance_name)
        print(f"Loaded {len(x_data)} data points")
        
        # Fit models
        results = fit_models(x_data, y_data)
        
        # Find best model
        best_model, best_r2 = find_best_model(results)
        
        # Print results
        print_results(results, best_model, best_r2)
        
        # Create plot
        plot_results(x_data, y_data, results, instance_name, save_plot)
        
        return {
            'instance_name': instance_name,
            'data_points': len(x_data),
            'results': results,
            'best_model': best_model,
            'best_r2_score': best_r2
        }
        
    except Exception as e:
        print(f"Error analyzing {instance_name}: {e}")
        return {'instance_name': instance_name, 'error': str(e)}

def print_summary(summary_file: str):
    """
    Print summary of results from a summary file.
    """
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    def compute_best_from_results(results_dict: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        valid: Dict[str, float] = {}
        for model_name, model_info in results_dict.items():
            if isinstance(model_info, dict) and 'r2_score' in model_info:
                r2 = model_info.get('r2_score')
                if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
                    try:
                        valid[model_name] = float(r2)
                    except Exception:
                        continue
        if not valid:
            return "None", None
        # Apply the same tie-breaking preference: linear gets +0.1 bonus
        selection_scores: Dict[str, float] = {}
        for model_name, r2 in valid.items():
            bonus = 0.1 if model_name == 'linear' else 0.0
            selection_scores[model_name] = r2 + bonus
        best_model_name = max(selection_scores, key=selection_scores.get)
        # Return raw R^2 for that best model
        return best_model_name, valid.get(best_model_name)
    
    # Build a compact summary: {instance: {best_model, best_r2_score}}
    summary_compact: Dict[str, Dict[str, Any]] = {}
    
    if isinstance(data, dict):
        # Expected format: {instance_name: {results: {...}, ...}, ...}
        for instance_name, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get('results'), dict):
                best_model, best_r2 = compute_best_from_results(entry['results'])
            elif isinstance(entry, dict):
                best_model, best_r2 = compute_best_from_results(entry)
            else:
                best_model, best_r2 = "None", None
            summary_compact[instance_name] = {
                'best_model': best_model,
                'best_r2_score': best_r2
            }
    elif isinstance(data, list):
        # Older format: list of per-instance dicts
        for entry in data:
            if not isinstance(entry, dict):
                continue
            instance_name = entry.get('instance_name') or entry.get('name') or 'unknown'
            results_dict = entry.get('results', {})
            if not isinstance(results_dict, dict):
                results_dict = {}
            best_model, best_r2 = compute_best_from_results(results_dict)
            summary_compact[instance_name] = {
                'best_model': best_model,
                'best_r2_score': best_r2
            }
    else:
        print("Unrecognized summary file format; nothing to summarize.")
        summary_compact = {}
    
    # Determine output path
    out_path = summary_file[:-5] + '_summary.json' if summary_file.endswith('.json') else summary_file + '_summary.json'
    with open(out_path, 'w') as f:
        json.dump(make_json_serializable(summary_compact), f, indent=2)
    print(f"Summary saved to: {out_path}")
    
    # Also save a CSV version
    out_csv_path = summary_file[:-5] + '_summary.csv' if summary_file.endswith('.json') else summary_file + '_summary.csv'
    with open(out_csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['instance_name', 'best_model', 'best_r2_score'])
        for instance_name, info in summary_compact.items():
            best_model = info.get('best_model', 'None')
            best_r2 = info.get('best_r2_score', '')
            writer.writerow([instance_name, best_model, best_r2 if best_r2 is not None else ''])
    print(f"CSV summary saved to: {out_csv_path}")

def main():
    """Main function to run regression analysis."""
    parser = argparse.ArgumentParser(description='Perform regression analysis on solving time data')
    parser.add_argument('--instance', '-i', type=str, help='Specific instance name to analyze')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all available instances')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting and saving plots')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--summary', '-s', type=str, help='Summary file for results')
    args = parser.parse_args()
    
    save_plot = not args.no_plot
    all_results = {}
    
    if args.summary:
        print_summary(args.summary)
        return

    if args.instance:
        # Analyze specific instance
        result = analyze_instance(args.instance, save_plot)
        all_results[args.instance] = result
        
    elif args.all:
        # Analyze all instances
        print("Analyzing all available instances...")
        for instance_name in get_all_instance_names():
            result = analyze_instance(instance_name, save_plot)
            all_results[instance_name] = result
            
    else:
        print("Please specify either --instance <name> or --all")
        return
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()