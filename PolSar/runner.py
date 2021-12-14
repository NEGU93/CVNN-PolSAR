import os
import sys
import json
import pathlib
from typing import Dict
from pdb import set_trace


def run_simulation(params: str):
    return f"python3 principal_simulation.py{params}"


def get_params(params_dict: Dict) -> str:
    result = ""
    for key, value in params_dict.items():
        if value:
            result += f" --{key}{f' {value}' if not isinstance(value, bool) else ''}"
    return result


def submit_job(job):
    with open('job.sh', 'w') as fp:
        fp.write(job)
    os.system("chmod +x job.sh")
    os.system("./job.sh")


if __name__ == "__main__":
    root_path = pathlib.Path(pathlib.Path(__file__).parent.resolve())
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = root_path / "ober_simulations.json"

    with open(str(json_path)) as json_file:
        config_json = json.load(json_file)

    for iterations in range(5):
        for param in config_json:
            # Launch the batch jobs
            param_str = get_params(param)
            submit_job(run_simulation(param_str))
