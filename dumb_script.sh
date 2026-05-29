#!/bin/bash                                                    
#SBATCH --time=0-4:0:0                                                      
#SBATCH --account=def-vganesh 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=./dumb_script_%j.log
# source .env
# source $PYENVPATH
# python src/scripts/Experiments/direct_regression_analysis.py --all --output regression.json 
git add -A; git commit -m "Update"; git push