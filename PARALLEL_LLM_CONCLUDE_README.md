# Parallel LLM Conclude Expression

This document describes the `parallel_llm_conclude_expression` function that has been implemented to process multiple instances in parallel for LLM-based equation analysis.

## Overview

The `parallel_llm_conclude_expression` function provides parallel processing capabilities for running LLM conclude expression analysis on multiple instances simultaneously. This significantly reduces the total processing time when analyzing large numbers of instances.

## Function Signature

```python
def parallel_llm_conclude_expression(
    instance_names=None, 
    use_cache=False, 
    config_type="auto", 
    plot=False, 
    max_workers=None, 
    batch_size=None
):
    """Run LLM conclude expression analysis on multiple instances in parallel
    
    Args:
        instance_names: List of instance names to process. If None, processes all instances.
        use_cache: Whether to use cached results
        config_type: PySR configuration type (not used in this function but kept for compatibility)
        plot: Whether to create comparison plots
        max_workers: Maximum number of parallel workers. If None, uses CPU count.
        batch_size: Number of instances to process in each batch. If None, processes all at once.
        
    Returns:
        Dict containing results for all processed instances
    """
```

## Features

### 1. Parallel Processing
- Uses Python's `multiprocessing.Pool` for true parallel execution
- Automatically determines optimal number of workers based on CPU count
- Supports custom worker count via `max_workers` parameter

### 2. Batch Processing
- Optional batch processing for memory management
- Processes instances in configurable batch sizes
- Useful for very large datasets or memory-constrained environments

### 3. Error Handling
- Comprehensive error handling for each instance
- Continues processing even if individual instances fail
- Detailed error reporting and categorization

### 4. Progress Tracking
- Real-time progress updates during processing
- Detailed statistics and timing information
- Comprehensive summary report

### 5. Caching Support
- Full integration with existing caching system
- Respects `use_cache` parameter for all instances
- Significantly speeds up repeated runs

### 6. Plot Generation
- Optional plot generation for each instance
- Creates both LLM-included and original-only plots
- Handles plot generation errors gracefully

## Usage Examples

### Basic Usage
```bash
# Process all instances in parallel
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression

# Process with caching enabled
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --use_cache

# Process with plots
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --plot
```

### Advanced Usage
```bash
# Process specific instances
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --instances 6s0 6s109 bob2

# Limit number of workers
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --max_workers 4

# Process in batches
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --batch_size 10

# Combine options
python -m src.scripts.Experiments.regression_analysis --parallel_llm_conclude_expression --use_cache --plot --max_workers 8 --batch_size 20
```

### Programmatic Usage
```python
from src.scripts.Experiments.regression_analysis import parallel_llm_conclude_expression

# Process all instances
results = parallel_llm_conclude_expression()

# Process specific instances with custom settings
results = parallel_llm_conclude_expression(
    instance_names=["6s0", "6s109", "bob2"],
    use_cache=True,
    plot=True,
    max_workers=4,
    batch_size=2
)
```

## Result Structure

The function returns a dictionary where keys are instance names and values are result dictionaries:

```python
{
    "instance_name": {
        "instance_name": "6s0",
        "status": "success",  # or "error", "insufficient_data", etc.
        "error": None,        # Error message if status is not "success"
        "timestamp": "2024-01-01T12:00:00",
        "conclusion": {       # Only present if status is "success"
            "llm_upper_bound": "x^2",
            "reasoning": "...",
            "type_of_equation": "polynomial",
            # ... other conclusion fields
        },
        "plot_error": "..."   # Only present if plot generation failed
    }
}
```

## Status Types

- `success`: Instance processed successfully
- `error`: General processing error
- `insufficient_data`: Not enough data points (≤5)
- `missing_sympy_summary`: Sympy summary file not found
- `invalid_sympy_data`: Invalid data in sympy summary
- `llm_failed`: LLM analysis failed

## Performance Considerations

### Memory Usage
- Each worker process loads its own copy of required modules
- Consider using `batch_size` for memory-constrained environments
- Monitor memory usage with large numbers of instances

### CPU Usage
- Uses all available CPU cores by default
- Use `max_workers` to limit CPU usage
- Consider system load when setting worker count

### I/O Considerations
- Each worker performs independent file I/O
- Caching significantly reduces repeated API calls
- Plot generation can be I/O intensive

## Output Files

### Summary Report
- Location: `results/parallel_llm_conclude_summary.json`
- Contains comprehensive statistics and all results
- Includes timing information and configuration details

### Individual Results
- Each instance's conclusion is saved to its respective cache file
- Plots are saved to `results/plots/` directory
- Follows existing naming conventions

## Error Handling

The function includes robust error handling:

1. **Instance-level errors**: Individual instance failures don't stop processing
2. **Worker errors**: Worker process crashes are handled gracefully
3. **Resource errors**: Memory and I/O errors are caught and reported
4. **API errors**: LLM API failures are handled with fallback options

## Dependencies

- `multiprocessing`: For parallel execution
- `json`: For data serialization
- `os`: For file system operations
- `time`: For timing measurements
- `datetime`: For timestamps

## Testing

A test script is provided (`test_parallel_llm.py`) that:
- Tests the function with a small subset of instances
- Verifies error handling
- Checks result structure
- Provides example usage

Run the test:
```bash
python test_parallel_llm.py
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `max_workers` or use `batch_size`
2. **API rate limits**: Enable caching with `--use_cache`
3. **Missing dependencies**: Ensure all required packages are installed
4. **File permissions**: Check write permissions for results directory

### Debug Mode

For debugging, you can modify the function to:
- Add more verbose logging
- Process fewer instances
- Use fewer workers
- Enable detailed error reporting

## Future Enhancements

Potential improvements for future versions:

1. **Progress bars**: Visual progress indicators
2. **Resume capability**: Resume interrupted processing
3. **Dynamic batching**: Automatic batch size adjustment
4. **Resource monitoring**: Real-time resource usage tracking
5. **Distributed processing**: Support for multiple machines
