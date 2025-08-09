import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ExperimentTemplate(ABC):
    """
    Template class for experiments with overridable input/output functions and progress saving.
    
    This class provides a framework for running experiments with:
    - Configurable input processing
    - Customizable output generation
    - Progress tracking and saving
    - Experiment state management
    """
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: str = "ExperimentResults",
                 save_interval: int = 1,
                 auto_save: bool = True):
        """
        Initialize the experiment template.
        
        Args:
            experiment_name: Name of the experiment for file naming
            output_dir: Directory to save experiment results
            save_interval: How often to save progress (every N iterations)
            auto_save: Whether to automatically save progress
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval
        self.auto_save = auto_save
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment state
        self.current_iteration = 0
        self.total_iterations = 0
        self.benchmark_dir = None
        self.log_dir = None
        self.results = []
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "initialized"
        }
        self.init_logging()
        self.init_benchmark_dir()
        # Load existing progress if available
        self._load_progress()
    
    def init_benchmark_dir(self):
        """
        Initialize the benchmark directory.
        """
        self.benchmark_dir = Path(self.output_dir / "benchmarks")
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        pass

    def init_logging(self):
        """
        Initialize logging.
        """
        self.log_dir = Path(self.output_dir / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> Any:
        """
        Configure the experiment. Override this method to implement custom configuration.
        """
        pass
    
    
    @abstractmethod
    def process_input(self, input_data: Any) -> Any:
        """
        Process the input data. Override this method to implement custom input processing.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed input data
        """
        pass
    
    @abstractmethod
    def generate_output(self, processed_input: Any) -> Any:
        """
        Generate output from processed input. Override this method to implement custom output generation.
        
        Args:
            processed_input: Processed input data
            
        Returns:
            Generated output
        """
        pass
    
    @abstractmethod
    def run(self):
        """
        Run the experiment.
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data. Override this method to implement custom validation.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output data. Override this method to implement custom validation.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        return True
    
    def run_single(self, input_data: Any) -> Dict[str, Any]:
        """
        Run a single experiment iteration.
        
        Args:
            input_data: Input data for the experiment
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Input validation failed")
            
            # Process input
            processed_input = self.process_input(input_data)
            
            # Generate output
            output_data = self.generate_output(processed_input)
            
            # Validate output
            if not self.validate_output(output_data):
                raise ValueError("Output validation failed")
            
            # Create result entry
            result = {
                "iteration": self.current_iteration,
                "input": input_data,
                "processed_input": processed_input,
                "output": output_data,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            self.results.append(result)
            self.current_iteration += 1
            
            # Auto-save if enabled
            if self.auto_save and self.current_iteration % self.save_interval == 0:
                self.save()
            
            return result
            
        except Exception as e:
            error_result = {
                "iteration": self.current_iteration,
                "input": input_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
            self.results.append(error_result)
            self.current_iteration += 1
            return error_result
    
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the current experiment progress.
        
        Args:
            filename: Optional custom filename for the save file
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_progress_{timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        # Update metadata
        self.metadata.update({
            "last_save": datetime.now().isoformat(),
            "current_iteration": self.current_iteration,
            "total_results": len(self.results),
            "status": "running"
        })
        
        # Prepare data to save
        save_data = {
            "metadata": self.metadata,
            "results": self.results,
            "current_iteration": self.current_iteration,
            "experiment_name": self.experiment_name
        }
        
        # Save using pickle for complex objects
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save a JSON version for human readability
        json_filepath = filepath.with_suffix('.json')
        json_data = {
            "metadata": self.metadata,
            "current_iteration": self.current_iteration,
            "experiment_name": self.experiment_name,
            "results_count": len(self.results)
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Progress saved to: {filepath}")
        return str(filepath)
    
    def _load_progress(self) -> bool:
        """
        Load existing progress from saved files.
        
        Returns:
            True if progress was loaded, False otherwise
        """
        # Look for the most recent save file
        pattern = f"{self.experiment_name}_progress_*.pkl"
        save_files = list(self.output_dir.glob(pattern))
        
        if not save_files:
            return False
        
        # Get the most recent file
        latest_file = max(save_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'rb') as f:
                save_data = pickle.load(f)
            
            self.metadata = save_data.get("metadata", self.metadata)
            self.results = save_data.get("results", [])
            self.current_iteration = save_data.get("current_iteration", 0)
            
            print(f"Loaded progress from: {latest_file}")
            print(f"Resuming from iteration: {self.current_iteration}")
            return True
            
        except Exception as e:
            print(f"Failed to load progress from {latest_file}: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment progress.
        
        Returns:
            Dictionary containing experiment summary
        """
        successful_results = [r for r in self.results if r.get("status") == "success"]
        error_results = [r for r in self.results if r.get("status") == "error"]
        
        return {
            "experiment_name": self.experiment_name,
            "current_iteration": self.current_iteration,
            "total_results": len(self.results),
            "successful_results": len(successful_results),
            "error_results": len(error_results),
            "success_rate": len(successful_results) / len(self.results) if self.results else 0,
            "start_time": self.metadata.get("start_time"),
            "last_save": self.metadata.get("last_save"),
            "status": self.metadata.get("status", "unknown")
        }
    
    def reset(self) -> None:
        """
        Reset the experiment state.
        """
        self.current_iteration = 0
        self.results = []
        self.metadata.update({
            "start_time": datetime.now().isoformat(),
            "status": "reset"
        })
        print("Experiment state reset")
    
    def finish(self) -> None:
        """
        Mark the experiment as finished and save final results.
        """
        self.metadata.update({
            "end_time": datetime.now().isoformat(),
            "status": "finished",
            "total_iterations": self.current_iteration
        })
        self.save(f"{self.experiment_name}_final.pkl")
        print("Experiment finished")