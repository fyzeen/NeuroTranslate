#!/bin/bash
#SBATCH --job-name=ICCs
#SBATCH --output=/home/ahmadf/batch/sbatch.out%j
#SBATCH --error=/home/ahmadf/batch/sbatch.err%j
#SBATCH --time=95:55:00
#SBATCH --mem=32GB
#SBATCH -N 1 #number nodes
#SBATCH -n 1 #number tasks per node
#SBATCH -w node13

module load python
source activate neurotranslate

python3 /home/ahmadf/NeuroTranslate/code/12_IntraClassCorrelations.py

conda deactivate
