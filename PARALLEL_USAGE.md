# Parallel LLM Conclude Expression Usage

## Overview

The `parallel_llm_conclude_expression` function has been implemented to process multiple instances in parallel by calling `run_pysr_and_conclude_with_llm` for each instance using multiple CPU cores.

## Basic Usage

### Process all instances in parallel:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression
```

### Process with caching enabled:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --use_cache
```

### Process with plots:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --plot
```

## Advanced Usage

### Process specific instances:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --instances 6s0 6s109 bob2
```

### Limit number of workers:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --max_workers 4
```

### Combine options:
```bash
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --use_cache --plot --max_workers 8 --instances 6s0 6s109
```

## Programmatic Usage

```python
from src.scripts.Experiments.regression_analysis import parallel_llm_conclude_expression

# Process all instances
results = parallel_llm_conclude_expression()

# Process specific instances with custom settings
results = parallel_llm_conclude_expression(
    instance_names=["6s0", "6s109", "bob2"],
    use_cache=True,
    plot=True,
    max_workers=4
)
```

## What it does

For each instance, the function:
1. Calls `run_pysr_and_conclude_with_llm(instance_name, use_cache, config_type, plot)`
2. This runs PySR regression analysis
3. Gets the sympy summary (equation and leading term)
4. Runs LLM conclude expression analysis
5. Creates comparison plots (if `plot=True`)

## Performance

- Uses all available CPU cores by default
- Each instance is processed independently in parallel
- Significantly faster than sequential processing
- Memory usage scales with number of workers

## Output

- Each instance's results are saved to their respective cache files
- Plots are saved to `results/plots/` directory
- Console output shows progress and summary statistics
- Function returns a dictionary with results for each instance
