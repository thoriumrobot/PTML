#!/bin/bash -l
#SBATCH --job-name=typeinfo
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --mem=8G

module purge
conda activate largemem
cd /home/k/ks225/annotations/2hv
python typeinfo.py >typeinfo.txt
