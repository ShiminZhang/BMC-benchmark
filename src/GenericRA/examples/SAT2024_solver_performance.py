from GenericRA import ExperimentTemplate
from typing import Dict, Any
import shutil
from pathlib import Path
import os

class SAT2024SolverPerformance(ExperimentTemplate):
    def __init__(self, experiment_name: str, benchmark_dir: str):
        super().__init__(experiment_name)
        self.copy_formulas_to_benchmark_dir(benchmark_dir)
        self.solver_map : Dict[str, str] = {} # solver_name -> solver_path
        

    def copy_formulas_to_benchmark_dir(self, external_benchmark_dir: str):
        """Copy the formulas to the benchmark directory."""
        external_path = Path(external_benchmark_dir)
        if not external_path.exists():
            raise FileNotFoundError(f"External benchmark directory not found: {external_benchmark_dir}")
        
        # Copy all .cnf files from external directory to benchmark directory
        for cnf_file in external_path.glob("*.cnf"):
            dest_path = self.benchmark_dir / cnf_file.name
            shutil.copy2(cnf_file, dest_path)
            print(f"Copied {cnf_file.name} to benchmark directory")
        
        print(f"Finished copying CNF files to {self.benchmark_dir}")
        pass

    def add_solver(self, solver_name: str, solver_path: str):
        """Add a solver to the experiment."""
        self.solver_map[solver_name] = solver_path
        pass

    def submit_slurm_job(self, solver_name: str, solver_path: str):
        assert solver_name in self.solver_map, f"Solver {solver_name} not found"
        assert solver_path == self.solver_map[solver_name], f"Solver {solver_path} not found"
        log_file = f"{self.log_dir}/{solver_name}.log"
        os.makedirs(f"{self.log_dir}/slurm", exist_ok=True)
        slurm_output_file = f"{self.log_dir}/slurm/{solver_name}.slurm.out"
        
        # Count CNF files to set array size
        cnf_files = list(self.benchmark_dir.glob("*.cnf"))
        array_size = len(cnf_files)
        assert array_size == 400
        # Create the wrapped command with proper SLURM array handling
        wrapped = f"#!/bin/bash\n"
        wrapped += f"instance_path=\"$(ls '{self.benchmark_dir}' | grep '.cnf' | sed -n ${{SLURM_ARRAY_TASK_ID}}p)\"\n"
        wrapped += f"{solver_path} \"{self.benchmark_dir}/$instance_path\" > {log_file} 2>&1"

        cmd = f"sbatch --mem=6g --array=1-{array_size} --time=01:30:00 --job-name=run_{solver_name} --output={log_file} --wrap=\"{wrapped}\""
        
        print(cmd)
        # os.system(cmd)

    def run(self):
        """Run the experiment."""

        for solver_name, solver_path in self.solver_map.items():
            


        pass