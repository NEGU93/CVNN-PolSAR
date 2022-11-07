#!/bin/sh

#SBATCH --job-name=testins_spiro
#SBATCH --ntasks=1
#SBATCH --time=2-2:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH --qos=co_long_gpu
#SBATCH --output=outlog/slurm.%j.out
#SBATCH --error=errlog/slurm.%j.err

cd $WORKDIR

module load anaconda/2020.11

conda activate tf-pip
    
mpirun python3 /scratchm/jbarrach/onera/PolSar/principal_simulation.py --dropout 0.3 0.3 None --coherency --epochs 1 --dataset_method random --model cao --dataset OBER

