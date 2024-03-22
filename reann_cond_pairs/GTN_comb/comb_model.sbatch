#!/bin/bash -l
#SBATCH --job-name=nullgtn
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module purge
conda activate largemem
module load CUDA
srun python gtn_preprocessing.py
