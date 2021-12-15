#!/bin/sh

#SBATCH --job-name=ober_real
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

python3 ../../principal_simulation.py --tensorflow --real_mode --coherency --epochs 400 --dataset_method random --model cao --dataset OBER


