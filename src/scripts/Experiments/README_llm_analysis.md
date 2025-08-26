# LLM Equation Analysis Module

This module provides LLM-powered analysis of regression equations to find closest upper bounds and analyze their computational complexity.

## Features

- **Multi-LLM Support**: Uses Gemini Pro (default) or GPT-4 to analyze mathematical equations and find asymptotically tight upper bounds
- **Upper Bound Analysis**: LLM-powered asymptotic analysis with mathematical reasoning
- **Complexity Analysis**: Analyzes the computational complexity of found upper bounds
- **Heuristic Fallback**: Built-in heuristic analysis when LLM is not available
- **Batch Processing**: Support for analyzing multiple equations at once
- **Error Handling**: Robust error handling with fallback parsing
- **Result Persistence**: Saves analysis results to JSON files

## Installation

1. Install required dependencies:
```bash
# For Gemini (recommended)
pip install google-generativeai

# For OpenAI (alternative)
pip install openai
```

2. Set your API key:
```bash
# For Gemini (recommended)
export GOOGLE_API_KEY='your-gemini-key-here'

# For OpenAI (alternative)
export OPENAI_API_KEY='your-openai-key-here'
```

## Usage

### Basic Usage

```python
from src.scripts.Experiments.llm_refit_curve import analyze_equation_with_llm

# Analyze a single equation (uses Gemini by default)
results = analyze_equation_with_llm("oc8051gm0caddr")
print(results)

# Specify provider and model
results = analyze_equation_with_llm("oc8051gm0caddr", provider="gemini", model="gemini-pro")
results = analyze_equation_with_llm("oc8051gm0caddr", provider="openai", model="gpt-4")
```

### Advanced Usage

```python
from src.scripts.Experiments.llm_refit_curve import (
    LLMEquationAnalyzer,
    batch_analyze_equations,
    print_analysis_summary
)

# Initialize analyzer with custom settings
analyzer = LLMEquationAnalyzer(api_key="your-key", model="gemini-pro", provider="gemini")

# Analyze specific equation
equation = "x0*x0/(x0 + (-1*1.7884021 + 53.146954/x0)*52.14305)"
upper_bound_result = analyzer.find_upper_bound(equation)
complexity_result = analyzer.analyze_complexity(upper_bound_result['upper_bound'])

# Batch processing
instance_names = ["instance1", "instance2", "instance3"]
batch_results = batch_analyze_equations(instance_names, provider="gemini")
print_analysis_summary(batch_results)
```

### Running the Demo

```bash
cd /path/to/BMC-benchmark
python -m src.scripts.Experiments.llm_refit_curve
```

## Example Output

```
Analyzing equation for instance: oc8051gm0caddr
==================================================

📊 ANALYSIS RESULTS
==================================================
🔍 Instance: oc8051gm0caddr
📈 Original Equation: x0*x0/(x0 + (-1*1.7884021 + 53.146954/x0)*52.14305)

🤖 LLM ANALYSIS (Using Gemini)
==============================

🎯 UPPER BOUND ANALYSIS
   Upper Bound: O(x0)
   Growth Rate: linear
   Asymptotic Notation: O(x0)
   Confidence: 0.95
   Reasoning: As x0 approaches infinity, the numerator grows as x0^2 while the denominator grows as x0*52.14305...

🔬 COMPLEXITY ANALYSIS
   Time Complexity: O(n)
   Space Complexity: O(1)
   Complexity Class: P
   Implications: Linear time complexity indicates efficient computation for large inputs...

💾 Results saved to: ./results/llm_analysis_oc8051gm0caddr.json
```

## API Reference

### `analyze_equation_with_llm(name, api_key=None, provider="gemini", model=None)`
Complete analysis pipeline for a given instance.

**Parameters:**
- `name` (str): Instance name to analyze
- `api_key` (str, optional): API key for the chosen provider
- `provider` (str): LLM provider ("gemini" or "openai")
- `model` (str, optional): Model to use (defaults to gemini-pro/gpt-4)

**Returns:**
- Dict containing complete analysis results

### `LLMEquationAnalyzer`
Main analyzer class for equation analysis.

**Methods:**
- `find_upper_bound(equation, variable="x0")`: Find upper bound for equation
- `analyze_complexity(upper_bound, variable="x0")`: Analyze complexity of upper bound

### `batch_analyze_equations(instance_names, api_key=None, provider="gemini", model=None)`
Analyze multiple equations in batch.

**Parameters:**
- `instance_names` (list): List of instance names
- `api_key` (str, optional): API key for the chosen provider
- `provider` (str): LLM provider ("gemini" or "openai")
- `model` (str, optional): Model to use for analysis

**Returns:**
- Dict mapping instance names to analysis results

## Configuration

The module uses the following default settings:
- **Default Provider**: Gemini (faster and often more accurate for mathematical analysis)
- **Models**: Gemini Pro (default) or GPT-4 (alternative)
- **Temperature**: 0.1 (low temperature for consistent mathematical analysis)
- **Max Tokens**: 1000 for upper bound analysis, 800 for complexity analysis

## Error Handling

The module includes comprehensive error handling:
- Missing Google AI or OpenAI package installation
- Invalid API keys for either provider
- Network connectivity issues
- JSON parsing failures (with fallback text parsing)
- Missing equation files
- Automatic fallback to heuristic analysis when LLM fails

## File Structure

```
results/
├── llm_analysis_<instance>.json     # Individual analysis results
├── llm_batch_analysis.json          # Batch analysis results
└── pysr_results/
    └── <instance>.summary.json      # Input equation files
```

## Notes

- The module expects equation files in PySR summary format
- Analysis quality depends on equation complexity and LLM model capabilities
- API costs apply for OpenAI usage
- Results are cached locally to avoid re-analysis
