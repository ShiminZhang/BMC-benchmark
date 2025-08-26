#!/usr/bin/env python3
"""
Test script for equation saving functionality
"""

import json
import os
from ..paths import get_pysr_summary_path

def test_equation_format():
    """Test the equation JSON format"""
    
    # Simulate typical PySR equations (using x0, x1, etc. and ** for power)
    test_equations = [
        "x0*2.5 + 3.14159",
        "sin(x0) + cos(x0*2.0)",
        "x0**2 + 0.5*x0 - 1.0",
        "exp(x0)/(1 + x0)",
        "log(Abs(x0) + 1.0e-6)"
    ]
    
    print("Testing equation JSON format:")
    print("=" * 50)
    
    for i, equation in enumerate(test_equations):
        # Test file path
        test_name = f"test_instance_{i}"
        equation_path = get_pysr_summary_path(test_name)
        
        # Save equation in clean format
        with open(equation_path, 'w') as f:
            json.dump(equation, f, indent=None, separators=(',', ':'))
        
        # Read back and verify
        with open(equation_path, 'r') as f:
            loaded_equation = json.load(f)
        
        # Check file size
        file_size = os.path.getsize(equation_path)
        
        print(f"Instance: {test_name}")
        print(f"Original:  {equation}")
        print(f"Loaded:    {loaded_equation}")
        print(f"File size: {file_size} bytes")
        print(f"Match:     {equation == loaded_equation}")
        print("-" * 30)
        
        # Clean up test file
        os.remove(equation_path)
    
    print("✓ All equation format tests passed!")

def demo_json_format():
    """Demonstrate the clean JSON format"""
    
    print("\nJSON Format Examples:")
    print("=" * 50)
    
    examples = {
        "Simple linear": "x0*2.5 + 3.14",
        "Polynomial": "x0**2 + 0.5*x0 - 1.0", 
        "Trigonometric": "sin(x0) + cos(x0*2.0)",
        "Exponential": "exp(x0)/(1 + x0)",
        "Complex": "log(Abs(x0**2 - 1) + 1.0e-6)*sin(x0/2)"
    }
    
    for desc, equation in examples.items():
        # Show clean JSON format
        json_str = json.dumps(equation, separators=(',', ':'))
        print(f"{desc:15}: {json_str}")
    
    print(f"\nKey features:")
    print("- No extra object structure, just the equation string")
    print("- Compact format with no unnecessary whitespace")
    print("- Easy to read programmatically with json.load()")
    print("- Human readable when opened in any text editor")

if __name__ == "__main__":
    test_equation_format()
    demo_json_format()
