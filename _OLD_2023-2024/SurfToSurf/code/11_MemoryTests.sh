#!/bin/bash
#SBATCH --job-name=MemTests
#SBATCH --output=/home/ahmadf/batch/sbatch.out%j
#SBATCH --error=/home/ahmadf/batch/sbatch.err%j
#SBATCH --time=47:55:00
#SBATCH --mem=32GB
#SBATCH -N 1 #number nodes
#SBATCH -n 1 #number tasks per node
#SBATCH --gres gpu:tesla_a100:1,vmem:40gb:1

module load python
source activate neurotranslate

python3 /home/ahmadf/NeuroTranslate/code/11_MemoryTests.py

conda deactivate

