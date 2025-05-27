#!/bin/bash
#SBATCH -J RegressionTests
#SBATCH -o /home/ahmadf/batch/sbatch.out%j
#SBATCH -e /home/ahmadf/batch/sbatch.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 10G 
#SBATCH --cpus-per-task 2
#SBATCH -t 0-24:00:00 

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

python3 regressions.py

conda activate