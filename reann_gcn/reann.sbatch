#!/bin/bash -l
#SBATCH --job-name=reann
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module purge
conda activate gcnenv
module load CUDA
cd /home/k/ks225/nullproj/reann_gcn/
mvn clean compile exec:java
