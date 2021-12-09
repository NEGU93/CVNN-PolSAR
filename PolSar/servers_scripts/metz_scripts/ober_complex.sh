#!/bin/sh

#SBATCH --job-name=ober_complex
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
    
python3 ../../principal_simulation.py --coherency --epochs 300 --dataset_method random --model cao --dataset OBER

