#!/bin/bash -l
#SBATCH --job-name=nullspec
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --mem=200G

module purge
conda activate spenv
srun python cluster_comb.py
