#!/bin/bash                                                    
#SBATCH --time=0-8:0:0                                                      
#SBATCH --account=def-vganesh 
#SBATCH --mem=16G
#SBATCH --output=./dumb_scheduler_%j.log

source ../general/bin/activate
# python scripts/prepare.py --prepare_sequential --pddef 1 --manage 
# python -m src.scripts.prepare_formulas --manage --time_limit 1800 --k_limit 5000 --step 1
python -m src.scripts.Experiments.collect_solving_time --all_slurm