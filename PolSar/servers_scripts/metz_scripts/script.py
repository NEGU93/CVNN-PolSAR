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
        # self.default_config_path = str(root_path / ("../../" + json_config_filename))
        os.makedirs(str(root_path / "logslurms"), exist_ok=True)  # Ensure the log directory exists

    def run_simulation(self, params: str):
        return f"""#!/bin/bash 
    
#SBATCH --job-name={self.name}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --mail-user=joseagustin.barra@gmail.com 
#SBATCH --mail-type=ALL
#SBATCH -e logslurms/slurm-%j.err
#SBATCH -o logslurms/slurm-%j.out

python3 -m pip install virtualenv --user
python3 -m virtualenv venv --system-site-packages
source venv/bin/activate
python3 -m pip install -r requirements.txt
    
python3 ../../principal_simulation.py{params}
"""

    @staticmethod
    def submit_job(job):
        with open('job.sh', 'w') as fp:
            fp.write(job)
        os.system("sbatch job.sh")
        # os.system("chmod +x job.sh")
        # os.system("./job.sh")


if __name__ == "__main__":
    MetzScheduler()()
