#!/bin/sh

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH --output=logs/outputJob.log
    
python3 ../../principal_simulation.py --coherency --epochs 200 --dataset_method random --model cao --dataset OBER --real_imag


