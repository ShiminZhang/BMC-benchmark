"""
Direct Regression Analysis Script

This script performs regression analysis on solving time data by fitting different models
(linear, polynomial, exponential) and comparing their performance using R-squared metrics.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Any, Optional

from ..paths import get_solving_times_path, get_plots_dir
from ..category import get_all_instance_names


def load_solving_data(instance_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load solving time data for a given instance.
    
    Args:
        instance_name: Name of the instance to load data for
        
    Returns:
        Tuple of (x_data, y_data) where x_data is CNF size and y_data is solving time
        
    Raises:
        FileNotFoundError: If the solving times file doesn't exist
        ValueError: If the data format is invalid
    """
    solving_times_path = get_solving_times_path(instance_name)
    
    if not os.path.exists(solving_times_path):
        raise FileNotFoundError(f"Solving times file not found: {solving_times_path}")
    
    try:
        with open(solving_times_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load solving times from {solving_times_path}: {e}")
    
    if not data:
        raise ValueError(f"No data found in {solving_times_path}")
    
    # Convert string keys to float and extract values
    x_data = [float(v["size_of_cnf"]) for v in data.values()]
    y_data = [float(v["solving_time"]) for v in data.values()]
    
    return np.array(x_data), np.array(y_data)


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
        poly_degree = min(4, len(x_data) - 1)
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
        # Normalize data to improve numerical stability
        x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data) + 1e-10)
        y_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data) + 1e-10)
        
        # Add bounds to prevent overflow and improve convergence
        # More conservative bounds for normalized data
        bounds = ([0.001, -5], [100, 5])
        
        # Better initial guess based on data characteristics
        if np.mean(y_norm) > 0.5:
            initial_guess = [1.0, 0.1]  # Growing exponential
        else:
            initial_guess = [1.0, -0.1]  # Decaying exponential
            
        popt_exp, pcov_exp = curve_fit(exponential_func, x_norm, y_norm, 
                                     p0=initial_guess, bounds=bounds, 
                                     maxfev=20000)
        
        # Transform back to original scale
        y_pred_norm = exponential_func(x_norm, *popt_exp)
        y_pred_exp = y_pred_norm * (np.max(y_data) - np.min(y_data)) + np.min(y_data)
        
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
            # Only try if all y values are positive
            if np.all(y_data > 0):
                log_y = np.log(y_data + 1e-10)  # Add small constant to avoid log(0)
                linear_coeffs = np.polyfit(x_data, log_y, 1)
                a = np.exp(linear_coeffs[1])
                b = linear_coeffs[0]
                y_pred_exp = a * np.exp(b * x_data)
                r2_exp = r2_score(y_data, y_pred_exp)
                
                results['exponential'] = {
                    'parameters': [a, b],
                    'covariance': None,
                    'predictions': y_pred_exp,
                    'r2_score': r2_exp
                }
                print(f"Alternative exponential model fitted with R² = {r2_exp:.4f}")
            else:
                raise ValueError("Cannot apply log transformation to non-positive values")
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
                   if 'r2_score' in v and not np.isnan(v['r2_score'])}
    
    if not valid_models:
        return "None", -np.inf
    
    best_model = max(valid_models, key=valid_models.get)
    return best_model, valid_models[best_model]


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


def main():
    """Main function to run regression analysis."""
    parser = argparse.ArgumentParser(description='Perform regression analysis on solving time data')
    parser.add_argument('--instance', '-i', type=str, help='Specific instance name to analyze')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all available instances')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting and saving plots')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    save_plot = not args.no_plot
    all_results = []
    
    if args.instance:
        # Analyze specific instance
        result = analyze_instance(args.instance, save_plot)
        all_results.append(result)
        
    elif args.all:
        # Analyze all instances
        print("Analyzing all available instances...")
        for instance_name in get_all_instance_names():
            result = analyze_instance(instance_name, save_plot)
            all_results.append(result)
            
    else:
        print("Please specify either --instance <name> or --all")
        return
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()