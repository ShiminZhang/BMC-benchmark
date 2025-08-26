import json
import os
import re
import math
from typing import Dict, Any, Tuple, Optional
from ..paths import get_pysr_summary_path, get_analysis_results_path, get_analysis_raw_output_path, get_solving_times_path, get_plot_path
from ..config import get_config_manager
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, lambdify, sympify

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai package not installed. Please install it with: pip install google-generativeai")
    genai = None
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai package not installed. Please install it with: pip install openai")
    openai = None
    OPENAI_AVAILABLE = False

def load_regression_equation(name):
    """Load regression equation from PySR summary file"""
    path = get_pysr_summary_path(name)
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, str):
        # Direct equation string
        return data
    elif isinstance(data, dict) and "equation" in data:
        # Dictionary with equation key
        return data["equation"]
    else:
        raise ValueError(f"Unknown data format in {path}: {type(data)}")


def heuristic_upper_bound_analysis(equation: str, variable: str = "x0") -> Dict[str, Any]:
    """Perform heuristic upper bound analysis without LLM
    
    Args:
        equation: Mathematical equation as string
        variable: Variable name (default: x0)
        
    Returns:
        Dict containing heuristic analysis results
    """
    result = {
        "upper_bound": "unknown",
        "reasoning": "",
        "confidence": 0.7,
        "growth_rate": "unknown",
        "asymptotic_notation": "unknown",
        "method": "heuristic"
    }
    
    reasoning = []
    
    # Count polynomial degrees in numerator and denominator
    numerator_degree = 0
    denominator_degree = 0
    
    # Simple pattern matching for polynomial terms
    import re
    
    # Split by division to separate numerator and denominator
    if "/" in equation:
        parts = equation.split("/", 1)
        numerator = parts[0].strip()
        denominator = parts[1].strip()
        
        # Count highest degree in numerator
        if f"{variable}*{variable}" in numerator:
            numerator_degree = max(numerator_degree, 2)
        elif variable in numerator:
            numerator_degree = max(numerator_degree, 1)
        
        # Count highest degree in denominator
        if f"{variable}*{variable}" in denominator:
            denominator_degree = max(denominator_degree, 2)
        elif variable in denominator:
            denominator_degree = max(denominator_degree, 1)
        
        reasoning.append(f"Numerator max degree: {numerator_degree}")
        reasoning.append(f"Denominator max degree: {denominator_degree}")
        
        # Determine asymptotic behavior
        if numerator_degree > denominator_degree:
            degree_diff = numerator_degree - denominator_degree
            if degree_diff == 1:
                result["upper_bound"] = f"{variable}"
                result["asymptotic_notation"] = f"O({variable})"
                result["growth_rate"] = "linear"
            elif degree_diff == 2:
                result["upper_bound"] = f"{variable}^2"
                result["asymptotic_notation"] = f"O({variable}^2)"
                result["growth_rate"] = "quadratic"
            else:
                result["upper_bound"] = f"{variable}^{degree_diff}"
                result["asymptotic_notation"] = f"O({variable}^{degree_diff})"
                result["growth_rate"] = f"polynomial degree {degree_diff}"
            
            reasoning.append(f"Rational function: degree {numerator_degree}/{denominator_degree}")
            reasoning.append(f"As {variable} → ∞, dominant behavior is O({variable}^{degree_diff})")
            
        elif numerator_degree == denominator_degree:
            result["upper_bound"] = "constant"
            result["asymptotic_notation"] = "O(1)"
            result["growth_rate"] = "constant"
            reasoning.append("Numerator and denominator have same degree")
            reasoning.append("Rational function approaches constant as x → ∞")
            
        else:
            result["upper_bound"] = f"1/{variable}^{denominator_degree - numerator_degree}"
            result["asymptotic_notation"] = f"O(1/{variable}^{denominator_degree - numerator_degree})"
            result["growth_rate"] = "decreasing"
            reasoning.append("Denominator degree higher than numerator")
            reasoning.append("Function approaches 0 as x → ∞")
    
    else:
        # Not a rational function, simple polynomial analysis
        if f"{variable}*{variable}" in equation:
            result["upper_bound"] = f"{variable}^2"
            result["asymptotic_notation"] = f"O({variable}^2)"
            result["growth_rate"] = "quadratic"
            reasoning.append("Detected quadratic terms")
        elif variable in equation:
            result["upper_bound"] = variable
            result["asymptotic_notation"] = f"O({variable})"
            result["growth_rate"] = "linear"
            reasoning.append("Detected linear terms")
        else:
            result["upper_bound"] = "constant"
            result["asymptotic_notation"] = "O(1)"
            result["growth_rate"] = "constant"
            reasoning.append("No variable terms detected")
    
    result["reasoning"] = ". ".join(reasoning)
    
    # Adjust confidence based on equation complexity
    complexity = equation.count('(') + equation.count('+') + equation.count('-')
    if complexity > 10:
        result["confidence"] = 0.5  # Lower confidence for complex equations
    elif complexity > 5:
        result["confidence"] = 0.6
    
    return result


class LLMEquationAnalyzer:
    """Analyzes regression equations using LLM to find upper bounds and complexity"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, provider: str = None):
        """Initialize the LLM analyzer
        
        Args:
            api_key: API key for the chosen provider (optional, will use config manager)
            model: Model to use for analysis (optional, will use config default)
            provider: LLM provider (optional, will use config default)
        """
        config_manager = get_config_manager()
        
        # Use config defaults if not specified
        if provider is None:
            provider = config_manager.get_default_provider()
        if model is None:
            model = config_manager.get_model_for_provider(provider)
        
        self.provider = provider
        self.model = model
        
        # Get API key from config manager if not provided
        if api_key is None:
            api_key = config_manager.get_api_key(provider)
        
        if not api_key:
            raise ValueError(f"No API key found for {provider}. Please set it using config manager or environment variables.")
        
        if provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI package not installed. Please install it with: pip install google-generativeai")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Please install it with: pip install openai")
            
            openai.api_key = api_key
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'gemini' or 'openai'")
    
    def find_upper_bound(self, equation: str, max_K: int, variable: str = "x0") -> Dict[str, Any]:
        """Use LLM to analyze equation and find asymptotic upper bound
        
        Args:
            equation: Mathematical equation as string
            variable: Variable name (default: x0)
            
        Returns:
            Dict containing upper_bound, reasoning, and confidence
        """
        # Clean the equation for better processing
        clean_equation = equation.replace("*", " * ").replace("/", " / ").replace("+", " + ").replace("-", " - ")
        
        prompt = f"""What is the asymptotic upper bound formula of {clean_equation} when K is large?

Please analyze the given equation,  give your answer of the upper bound formula, the formula should either be an exponential function or a polynomial function. explain why it is the upper bound, analyze the complexity of the upper bound formula. You must provide all your answer in following JSON format!!!:
{{
    "thinking": "",
    "reasoning": "",
    "upper_bound": "x + 5",
    "confidence": 0.9,
    "complexity": "O(x)"
}}"""
        
        try:
            if self.provider == "gemini":
                # Use Gemini API
                system_prompt = "You are a math expert in complexity theory."
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1
                    )
                )
                print(f"Gemini response: {response}")
                # Handle Gemini response safely
                if response.candidates and response.candidates[0].content.parts:
                    # Extract text from all parts
                    content_parts = []
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            content_parts.append(part.text)
                    content = ''.join(content_parts).strip()
                    
                    if not content:
                        raise Exception("Gemini returned empty content in response parts")
                else:
                    # Handle blocked or empty responses
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        reason = response.prompt_feedback.block_reason
                        raise Exception(f"Gemini blocked the request: {reason}")
                    else:
                        raise Exception("Gemini returned empty response - possibly due to safety filters")
                
            elif self.provider == "openai":
                # Use OpenAI API
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a mathematical analysis expert specializing in asymptotic analysis and complexity theory."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Try to parse JSON response
            try:
                # First try direct JSON parsing
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1)
                        # Clean up the JSON string - remove extra whitespace and fix newlines
                        json_str = re.sub(r'\n\s*', ' ', json_str)  # Replace newlines with spaces
                        json_str = re.sub(r'\s+', ' ', json_str)    # Normalize whitespace
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse extracted JSON: {e}")
                        # Try a more aggressive approach - find the JSON structure
                        try:
                            # Look for the complete JSON structure manually
                            start_idx = content.find('```json')
                            end_idx = content.find('```', start_idx + 7)
                            if start_idx != -1 and end_idx != -1:
                                json_content = content[start_idx + 7:end_idx].strip()
                                result = json.loads(json_content)
                                return result
                        except json.JSONDecodeError:
                            pass
                
                print(f"JSON parsing failed: {content}, try to parse manually")
                # If JSON parsing fails, extract information manually
                return self._parse_llm_response(content)
                
        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "upper_bound": None,
                "reasoning": None,
                "confidence": 0.0,
                "growth_rate": "unknown"
            }
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse non-JSON LLM response for upper bound analysis"""
        result = {
            "upper_bound": None,
            "reasoning": content,
            "confidence": 0.5,
            "growth_rate": "unknown",
            "asymptotic_notation": "unknown"
        }
        
        # Try to extract key information using regex
        upper_bound_match = re.search(r'upper bound[:\s]*([^.\n]+)', content, re.IGNORECASE)
        if upper_bound_match:
            result["upper_bound"] = upper_bound_match.group(1).strip()
        
        confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', content, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = float(confidence_match.group(1))
        
        return result
    
    def _parse_complexity_response(self, content: str) -> Dict[str, Any]:
        """Parse non-JSON LLM response for complexity analysis"""
        result = {
            "time_complexity": "unknown",
            "space_complexity": "unknown", 
            "complexity_class": "unknown",
            "practical_implications": content
        }
        
        # Try to extract complexity information
        time_complexity_match = re.search(r'O\([^)]+\)', content)
        if time_complexity_match:
            result["time_complexity"] = time_complexity_match.group(0)
        
        return result


def analyze_equation_with_llm(name: str, api_key: Optional[str] = None, provider: str = None, model: str = None) -> Dict[str, Any]:
    """Complete analysis pipeline: load equation, find upper bound, analyze complexity
    
    Args:
        name: Instance name to analyze
        api_key: API key for the chosen provider (optional, will use config manager)
        provider: LLM provider (optional, will use config default)
        model: Model to use (optional, will use config default)
        
    Returns:
        Complete analysis results
    """
    try:
        # Load the regression equation
        equation = load_regression_equation(name)
        print(f"Loaded equation for {name}: {equation}")
        
        # Initialize LLM analyzer (will use config manager for defaults)
        analyzer = LLMEquationAnalyzer(api_key=api_key, model=model, provider=provider)
        x_data, y_data = load_original_data(name)
        max_K = x_data[-1]
        
        # Find upper bound
        print("Finding upper bound with LLM...")
        upper_bound_analysis = analyzer.find_upper_bound(equation, max_K)
        
        if "error" in upper_bound_analysis:
            return {
                "instance_name": name,
                "original_equation": equation,
                "error": upper_bound_analysis["error"]
            }
        
        # print(f"Upper bound found: {upper_bound_analysis.get('upper_bound', 'N/A')}")
        
        # # Analyze complexity of the upper bound
        # if upper_bound_analysis.get("upper_bound"):
        #     print("Analyzing complexity...")
        #     complexity_analysis = analyzer.analyze_complexity(upper_bound_analysis["upper_bound"])
        # else:
        #     complexity_analysis = {"error": "No upper bound to analyze"}
        
        # Combine results
        # result = {
        #     "instance_name": name,
        #     "original_equation": equation,
        #     "upper_bound_analysis": upper_bound_analysis,
        #     "complexity_analysis": complexity_analysis
        # }
        upper_bound_analysis["instance_name"] = name
        upper_bound_analysis["original_equation"] = equation
        return upper_bound_analysis
        
    except Exception as e:
        return {
            "instance_name": name,
            "error": f"Analysis failed: {str(e)}"
        }


def analyze_equation_heuristic(name: str) -> Dict[str, Any]:
    """Complete heuristic analysis pipeline without LLM
    
    Args:
        name: Instance name to analyze
        
    Returns:
        Complete heuristic analysis results
    """
    try:
        # Load the regression equation
        equation = load_regression_equation(name)
        
        # Perform heuristic analysis
        upper_bound_analysis = heuristic_upper_bound_analysis(equation)
        
        # Simple complexity mapping based on growth rate
        complexity_analysis = {
            "time_complexity": upper_bound_analysis["asymptotic_notation"],
            "space_complexity": "O(1)",  # Assume constant space for equation evaluation
            "complexity_class": "P" if upper_bound_analysis["growth_rate"] in ["constant", "linear", "quadratic"] else "unknown",
            "method": "heuristic_mapping"
        }
        
        # Map growth rates to practical implications
        growth_rate = upper_bound_analysis["growth_rate"]
        if growth_rate == "constant":
            complexity_analysis["practical_implications"] = "Very efficient - constant time regardless of input size"
        elif growth_rate == "linear":
            complexity_analysis["practical_implications"] = "Efficient - scales linearly with input size"
        elif growth_rate == "quadratic":
            complexity_analysis["practical_implications"] = "Moderate - may become slow for very large inputs"
        else:
            complexity_analysis["practical_implications"] = "Performance depends on specific growth pattern"
        
        # Combine results
        result = {
            "instance_name": name,
            "original_equation": equation,
            "upper_bound_analysis": upper_bound_analysis,
            "complexity_analysis": complexity_analysis,
            "analysis_method": "heuristic"
        }
        
        return result
        
    except Exception as e:
        return {
            "instance_name": name,
            "error": f"Heuristic analysis failed: {str(e)}",
            "analysis_method": "heuristic"
        }

def llm_analysis(instance_name, use_cache=False):

    print(f"Analyzing equation for instance: {instance_name}")
    print("=" * 50)
    
    # Check if API keys are available via config manager
    config_manager = get_config_manager()
    
    # Try to get API key for default provider
    default_provider = config_manager.get_default_provider()
    api_key = config_manager.get_api_key(default_provider)
    
    # If default provider doesn't have key, try the other one
    if not api_key:
        other_provider = "openai" if default_provider == "gemini" else "gemini"
        api_key = config_manager.get_api_key(other_provider)
        if api_key:
            default_provider = other_provider
    
    if not api_key:
        print(f"\n🤖 LLM ANALYSIS")
        print("=" * 20)
        print("⚠️  No LLM API keys found.")
        print("To set up API keys, run:")
        print("python -m src.scripts.config")
        print("\nOr set environment variables:")
        print("For Gemini: export GOOGLE_API_KEY='your-gemini-key-here'")
        print("For OpenAI: export OPENAI_API_KEY='your-openai-key-here'")
        print("\nAlternatively, you can call:")
        print("analyze_equation_with_llm('instance_name', api_key='your-key', provider='gemini')")
        return
    
    print(f"\n🤖 LLM ANALYSIS (Using {default_provider.title()})")
    print("=" * 35)
    
    try:

        # Run the complete analysis
        # results = analyze_equation_with_llm(instance_name, api_key=api_key, provider=default_provider)
        cache_path = get_analysis_raw_output_path(instance_name)
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from cache: {cache_path}")
        else:
            results = analyze_equation_with_llm(instance_name, api_key=api_key, provider=default_provider)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results["timestamp"] = timestamp
            with open(cache_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to cache: {cache_path}")

        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"🔍 Instance: {instance_name}")
        print(f"📈 Original Equation: {results['original_equation']}")
        
        # Upper bound analysis
        ub_analysis = results
        print(f"\n🎯 UPPER BOUND ANALYSIS")
        print(f"   Upper Bound: {ub_analysis.get('upper_bound', 'N/A')}")
        print(f"   Reason: {ub_analysis.get('reasoning', 'N/A')}")
        print(f"   Complexity: {ub_analysis.get('complexity', 'N/A')}")
        print(f"   Confidence: {ub_analysis.get('confidence', 'N/A')}")
        
        # Save results to file
        output_file = get_analysis_results_path(instance_name)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("Please check your API key and network connection.")

def load_original_data(instance_name):
    """Load original data points from solving times JSON"""
    solving_times_path = get_solving_times_path(instance_name)
    
    if not os.path.exists(solving_times_path):
        raise FileNotFoundError(f"Solving times file not found: {solving_times_path}")
    
    with open(solving_times_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to float and extract values
    x_data = [float(k) for k in data.keys()]
    y_data = [float(v) for v in data.values()]
    
    return np.array(x_data), np.array(y_data)


def load_llm_analysis(instance_name):
    """Load LLM analysis results"""
    analysis_path = get_analysis_results_path(instance_name)
    
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"LLM analysis file not found: {analysis_path}")
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    return analysis


def safe_evaluate_equation(equation_str, x_values, variable_name='x0', is_llm_equation=False):
    """Safely evaluate a mathematical equation over a range of x values"""
    try:
        # Parse the equation using sympy
        x = symbols(variable_name)
        
        if is_llm_equation:
            # Use special parsing for LLM equations
            equation = parse_llm_upper_bound(equation_str)
            if equation is None:
                return np.full_like(x_values, np.nan)
        else:
            equation = sympify(equation_str)
        
        # Convert to a lambda function for fast evaluation
        func = lambdify(x, equation, 'numpy')
        
        # Evaluate the function with error handling for individual points
        y_values = np.full_like(x_values, np.nan)
        
        for i, x_val in enumerate(x_values):
            try:
                result = func(x_val)
                if np.isscalar(result):
                    if np.isfinite(result) and not np.iscomplex(result):
                        y_values[i] = float(result)
                    elif np.iscomplex(result):
                        real_part = np.real(result)
                        if np.isfinite(real_part):
                            y_values[i] = real_part
                else:
                    # Handle array results
                    if np.isfinite(result) and not np.iscomplex(result):
                        y_values[i] = float(result)
            except (ZeroDivisionError, ValueError, TypeError, OverflowError):
                # Keep as NaN for problematic points
                continue
        
        return y_values
        
    except Exception as e:
        print(f"Error evaluating equation '{equation_str}': {e}")
        return np.full_like(x_values, np.nan)


def parse_llm_upper_bound(upper_bound_str):
    """Parse LLM upper bound string to a sympy-compatible equation"""
    if not upper_bound_str:
        return None
    
    try:
        # Handle common formats from LLM responses
        cleaned = upper_bound_str.strip()
        
        # Replace |x0| with Abs(x0) for sympy compatibility
        cleaned = cleaned.replace('|x0|', 'Abs(x0)')
        
        # Replace x^n with x**n for sympy compatibility
        cleaned = re.sub(r'\^(\d+)', r'**\1', cleaned)
        
        # Handle scientific notation if present
        cleaned = re.sub(r'(\d+\.?\d*)[eE]([+-]?\d+)', r'\1*10**\2', cleaned)
        
        # Parse with sympy
        return sympify(cleaned)
        
    except Exception as e:
        print(f"Error parsing LLM upper bound '{upper_bound_str}': {e}")
        return None


def plot_original_vs_llm_results(instance_name):
    """
    Plot comparison of original data, PySR equation, and LLM upper bound
    
    This function creates a comprehensive visualization showing:
    - Original data points (black scatter plot)
    - PySR regression equation (blue solid line)
    - LLM-suggested upper bound (red dashed line)
    
    The plot is saved to results/plots/{instance_name}.png
    
    Args:
        instance_name (str): Name of the instance to analyze and plot
        
    Returns:
        str: Path to the saved plot file, or None if plotting failed
        
    Raises:
        FileNotFoundError: If required data files are missing
        
    Example:
        >>> plot_path = plot_original_vs_llm_results("6s0")
        >>> print(f"Plot saved to: {plot_path}")
    """
    try:
        print(f"Creating comparison plot for {instance_name}...")
        
        # Load original data points
        x_data, y_data = load_original_data(instance_name)
        print(f"Loaded {len(x_data)} data points")
        
        # Load original PySR equation
        original_equation = load_regression_equation(instance_name)
        print(f"Original PySR equation: {original_equation}")
        
        # Load LLM analysis
        llm_analysis = load_llm_analysis(instance_name)
        llm_upper_bound = llm_analysis.get('upper_bound', '')
        print(f"LLM upper bound: {llm_upper_bound}")
        
        # Create x range for plotting equations (extend beyond data range)
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min
        x_plot = np.linspace(max(1, x_min - 0.1 * x_range), x_max + 0.1 * x_range, 1000)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot original data points
        plt.scatter(x_data, y_data, color='black', alpha=0.7, s=50, 
                   label='Original Data', zorder=5)
        
        # Plot original PySR equation
        y_original = safe_evaluate_equation(original_equation, x_plot)
        valid_mask = ~np.isnan(y_original) & np.isfinite(y_original)
        if np.any(valid_mask):
            # Filter out extreme values that might cause plotting issues
            y_plot_max = y_data.max() * 2  # Reasonable upper limit
            reasonable_mask = valid_mask & (y_original <= y_plot_max) & (y_original >= -y_plot_max)
            
            if np.any(reasonable_mask):
                plt.plot(x_plot[reasonable_mask], y_original[reasonable_mask], 
                        color='blue', linewidth=2, label='PySR Equation', alpha=0.8)
        
        # Plot LLM upper bound if available
        if llm_upper_bound:
            try:
                # Check if it's a simple constant first
                try:
                    constant_value = float(llm_upper_bound)
                    # It's a constant - plot as horizontal line
                    plt.axhline(y=constant_value, color='red', linewidth=2, 
                              linestyle='--', label=f'LLM Upper Bound ({constant_value:.1f})', 
                              alpha=0.8)
                except ValueError:
                    # It's not a simple constant, evaluate as equation
                    y_llm = safe_evaluate_equation(llm_upper_bound, x_plot, is_llm_equation=True)
                    valid_mask_llm = ~np.isnan(y_llm) & np.isfinite(y_llm)
                    
                    if np.any(valid_mask_llm):
                        # Filter out extreme values
                        y_plot_max = y_data.max() * 2
                        reasonable_mask_llm = valid_mask_llm & (y_llm <= y_plot_max) & (y_llm >= -y_plot_max)
                        
                        if np.any(reasonable_mask_llm):
                            plt.plot(x_plot[reasonable_mask_llm], y_llm[reasonable_mask_llm], 
                                    color='red', linewidth=2, linestyle='--', 
                                    label='LLM Upper Bound', alpha=0.8)
            except Exception as e:
                print(f"Could not plot LLM upper bound: {e}")
        
        # Formatting
        plt.xlabel('K (Input Size)', fontsize=12)
        plt.ylabel('Solving Time (seconds)', fontsize=12)
        plt.title(f'Comparison: Original Data vs Equations\nInstance: {instance_name}', 
                 fontsize=14, pad=20)
        
        # Clean up legend to avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
        
        plt.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits (avoid extreme outliers in equations)
        y_data_max = y_data.max()
        y_data_min = y_data.min()
        plt.ylim(max(0, y_data_min - 0.1 * (y_data_max - y_data_min)), 
                y_data_max * 1.2)
        
        # Add text box with equation info
        info_text = f"PySR: {original_equation[:60]}{'...' if len(original_equation) > 60 else ''}\n"
        if llm_upper_bound:
            info_text += f"LLM: {llm_upper_bound[:60]}{'...' if len(llm_upper_bound) > 60 else ''}"
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the plot
        plot_path = get_plot_path(instance_name)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {plot_path}")
        
        # Close the plot to free memory and avoid display issues
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating plot for {instance_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze equations with LLM")
    parser.add_argument("--instance_name", type=str, default="oc8051gm0caddr", help="Instance name to analyze")
    parser.add_argument("--use_cache", action="store_true", help="Use cache")
    parser.add_argument("--plot", action="store_true", help="Create comparison plot")
    args = parser.parse_args()
    instance_name = args.instance_name
    use_cache = args.use_cache
    
    # Run LLM analysis
    llm_analysis(instance_name, use_cache)
    
    # Create plot if requested
    if args.plot:
        print("\n📊 Creating comparison plot...")
        plot_path = plot_original_vs_llm_results(instance_name)
        if plot_path:
            print(f"✅ Plot saved to: {plot_path}")
        else:
            print("❌ Plot creation failed")


# Additional utility functions for batch processing
def batch_analyze_equations(instance_names: list, api_key: Optional[str] = None, provider: str = None, model: str = None) -> Dict[str, Any]:
    """Analyze multiple equations in batch
    
    Args:
        instance_names: List of instance names to analyze
        api_key: API key for the chosen provider
        provider: LLM provider ("gemini" or "openai")
        model: Model to use for analysis
        
    Returns:
        Dict mapping instance names to analysis results
    """
    results = {}
    
    for name in instance_names:
        print(f"\nAnalyzing {name}...")
        try:
            results[name] = analyze_equation_with_llm(name, api_key, provider, model)
        except Exception as e:
            results[name] = {"error": f"Failed to analyze {name}: {str(e)}"}
    
    return results


def save_analysis_results(results: Dict[str, Any], output_file: str = None):
    """Save analysis results to JSON file
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path (optional)
    """
    if output_file is None:
        output_file = "./results/llm_batch_analysis.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def print_analysis_summary(results: Dict[str, Any]):
    """Print a summary of analysis results
    
    Args:
        results: Analysis results from batch_analyze_equations
    """
    print("\n" + "="*60)
    print("📊 BATCH ANALYSIS SUMMARY")
    print("="*60)
    
    for instance_name, result in results.items():
        print(f"\n🔍 {instance_name}:")
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        # Upper bound info
        if "upper_bound_analysis" in result:
            ub = result["upper_bound_analysis"]
            print(f"   🎯 Upper Bound: {ub.get('upper_bound', 'N/A')}")
            print(f"   📈 Growth Rate: {ub.get('growth_rate', 'N/A')}")
        
        # Complexity info  
        if "complexity_analysis" in result:
            comp = result["complexity_analysis"]
            print(f"   ⏱️  Time Complexity: {comp.get('time_complexity', 'N/A')}")
            print(f"   🏷️  Complexity Class: {comp.get('complexity_class', 'N/A')}")

if __name__ == "__main__":
    main()