"""
GenericRA - A Generic Experiment Framework

This module provides a flexible framework for running experiments with:
- Configurable input processing
- Customizable output generation
- Progress tracking and saving
- Experiment state management
- Logging utilities

Main Classes:
    - ExperimentTemplate: Abstract base class for experiments
    - SimpleExperiment: Basic experiment implementation
    - TextProcessingExperiment: Text-specific experiment implementation

Logging Functions:
    - LOG: Basic logging function
    - LOG_TAG: Tagged logging function
    - TOGGLE_SHOWLOG: Enable/disable logging
    - REG_TAG: Register visible tags for logging
"""

# Import main experiment classes
from .src.experiment import (
    ExperimentTemplate,
)

# Import logging utilities
from .src.logging import (
    LOG,
    LOG_TAG,
    TOGGLE_SHOWLOG,
    REG_TAG,
    Showlog,
    VISIBLE_TAGS
)

# Version information
__version__ = "1.0.0"
__author__ = "GenericRA Team"

# Main exports
__all__ = [
    # Experiment classes
    "ExperimentTemplate",
    
    # Logging functions
    "LOG",
    "LOG_TAG", 
    "TOGGLE_SHOWLOG",
    "REG_TAG",
    
    # Logging state
    "Showlog",
    "VISIBLE_TAGS",
    
    # Version info
    "__version__",
    "__author__"
]
