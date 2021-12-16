import os
import sys
import json
import pathlib
from typing import List, Dict, Iterable
from pdb import set_trace

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

def get_params(params_dict: Dict) -> str:
    result = ""
    for key, value in params_dict.items():
        if value:
            result += f" --{key}{f' {value}' if not isinstance(value, bool) else ''}"
    return result

def submit_job(job):
    with open('job.sbatch', 'w') as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


if __name__ == "__main__":
    root_path = pathlib.Path(pathlib.Path(__file__).parent.resolve())
    # Ensure the log directory exists
    os.makedirs(str(root_path / "logslurms"), exist_ok=True)
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = root_path / "ober_simulations.json"

    with open(str(json_path)) as json_file:
        config_json = json.load(json_file)

    for iterations in range(1):
        for param in config_json:
            # Launch the batch jobs
            param_str = get_params(param)
            submit_job(run_simulation(param_str))
