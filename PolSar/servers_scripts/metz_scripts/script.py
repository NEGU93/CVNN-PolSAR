import os
import sys
import json
import pathlib
import argparse
sys.path.insert(1, "../../")
from runner import SimulationScheduler
from pdb import set_trace


class MetzScheduler(SimulationScheduler):

    def __init__(self):
        super().__init__()
        root_path = pathlib.Path(pathlib.Path(__file__).parent.resolve())
        self.default_config_path = str(root_path / "../../ober_simulations.json")
        os.makedirs(str(root_path / "logslurms"), exist_ok=True)  # Ensure the log directory exists

    @staticmethod
    def run_simulation(params : str):
        return f"""#!/bin/bash 
    
#SBATCH --job-name=ober
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH -e logslurms/slurm-%j.err
#SBATCH -o logslurms/slurm-%j.out
    
python3 ../../principal_simulation.py{params}
"""

    @staticmethod
    def submit_job(job):
        with open('job.sbatch', 'w') as fp:
            fp.write(job)
        os.system("sbatch job.sbatch")


if __name__ == "__main__":
    MetzScheduler()()
