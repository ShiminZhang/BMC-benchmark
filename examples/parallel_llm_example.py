#!/usr/bin/env python3
"""
Example usage of parallel_llm_conclude_expression function
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def example_basic_usage():
    """Example of basic parallel processing"""
    from scripts.Experiments.regression_analysis import parallel_llm_conclude_expression
    
    print("🚀 Example: Basic parallel processing")
    print("=" * 50)
    
    # Process all instances with default settings
    results = parallel_llm_conclude_expression(
        use_cache=True,  # Use cached results when available
        plot=False       # Don't create plots for this example
    )
    
    print(f"✅ Processed {len(results)} instances")
    return results

def example_custom_instances():
    """Example with specific instances"""
    from scripts.Experiments.regression_analysis import parallel_llm_conclude_expression
    
    print("\n🎯 Example: Custom instance list")
    print("=" * 50)
    
    # Process specific instances
    custom_instances = ["6s0", "6s109", "bob2"]
    results = parallel_llm_conclude_expression(
        instance_names=custom_instances,
        use_cache=True,
        plot=True,        # Create plots for these instances
        max_workers=2     # Use only 2 workers
    )
    
    print(f"✅ Processed {len(results)} custom instances")
    return results

def example_batch_processing():
    """Example with batch processing"""
    from scripts.Experiments.regression_analysis import parallel_llm_conclude_expression
    
    print("\n📦 Example: Batch processing")
    print("=" * 50)
    
    # Process in batches for memory efficiency
    results = parallel_llm_conclude_expression(
        use_cache=True,
        plot=False,
        max_workers=4,    # Use 4 workers
        batch_size=10     # Process 10 instances per batch
    )
    
    print(f"✅ Processed {len(results)} instances in batches")
    return results

def analyze_results(results):
    """Analyze and display results summary"""
    print("\n📊 Results Analysis")
    print("=" * 50)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    insufficient_data_count = sum(1 for r in results.values() if r["status"] == "insufficient_data")
    
    print(f"Total instances: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Insufficient data: {insufficient_data_count}")
    
    # Show some successful results
    successful_results = [r for r in results.values() if r["status"] == "success"]
    if successful_results:
        print(f"\n✅ Sample successful results:")
        for i, result in enumerate(successful_results[:3]):  # Show first 3
            conclusion = result.get("conclusion", {})
            print(f"  {result['instance_name']}: {conclusion.get('llm_upper_bound', 'N/A')}")
    
    # Show some errors
    error_results = [r for r in results.values() if r["status"] == "error"]
    if error_results:
        print(f"\n❌ Sample errors:")
        for i, result in enumerate(error_results[:3]):  # Show first 3
            print(f"  {result['instance_name']}: {result.get('error', 'Unknown error')}")

def main():
    """Main example function"""
    print("🧪 Parallel LLM Conclude Expression Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic usage
        results1 = example_basic_usage()
        analyze_results(results1)
        
        # Example 2: Custom instances
        results2 = example_custom_instances()
        analyze_results(results2)
        
        # Example 3: Batch processing
        results3 = example_batch_processing()
        analyze_results(results3)
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
