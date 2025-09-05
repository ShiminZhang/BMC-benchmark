#!/usr/bin/env python3
"""
Test script to verify the plotting functions work correctly
"""

import sys
import os
sys.path.append('src/scripts')

from Experiments.llm_refit_curve import plot_original_vs_llm_results

def test_plotting():
    """Test the plotting function with a sample instance"""
    # Test with instance 6s0 which has analysis results
    instance_name = "6s0"
    
    print(f"Testing plotting function with instance: {instance_name}")
    
    try:
        plot_path = plot_original_vs_llm_results(instance_name)
        if plot_path:
            print(f"✅ Plot created successfully: {plot_path}")
            return True
        else:
            print("❌ Plot creation failed")
            return False
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        return False

if __name__ == "__main__":
    success = test_plotting()
    sys.exit(0 if success else 1)
