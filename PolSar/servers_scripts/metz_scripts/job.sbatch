#!/bin/bash 
    
#SBATCH --job-name=ober
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH -e logslurms/slurm-%j.err
#SBATCH -o logslurms/slurm-%j.out
    
python3 ../../principal_simulation.py --model tan --epochs 100 --balance None --coherency --dataset OBER --tensorflow --real_mode
