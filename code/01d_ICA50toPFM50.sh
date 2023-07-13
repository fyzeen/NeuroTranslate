#!/bin/bash
#SBATCH --job-name=TestConvTrain
#SBATCH --output=/home/ahmadf/batch/sbatch.out%j
#SBATCH --error=/home/ahmadf/batch/sbatch.err%j
#SBATCH --time=23:55:00
#SBATCH --mem=32GB
#SBATCH -N 1 #number nodes
#SBATCH -n 1 #number tasks per node
#SBATCH --gres gpu:1,vmem:32gb:1


module load python
source activate neurotranslate

python3 /home/ahmadf/NeuroTranslate/code/01d_ICA50toPFM50.py

conda deactivate

