import os
import sys
import json
import pathlib
from typing import Dict, List
import argparse
from collections.abc import Iterable
from abc import abstractmethod, ABC
from pdb import set_trace


def add_to_all(key: str, value, list_dict):
    for i in range(len(list_dict)):
        list_dict[i][key] = value
    return list_dict


def add_constants(dictionary: Dict) -> List[Dict]:
    return_list = []
    constant = {"coherency": True, "dataset": "OBER"}
    variables = [{}, {"tensorflow": True, "real_mode": True}]
    dictionary.update(constant)
    for var in variables:
        return_list.append(dict(dictionary, **var))
    return return_list


class SimulationScheduler(ABC):

    def __init__(self):
        self.name = "simulation scheduler"  # json_config_filename.split('_')[0]

    def run_simulation(self, params: str):
        raise NotImplementedError(f"Need to implement this function first")

    @staticmethod
    def submit_job(job):
        raise NotImplementedError(f"Need to implement this function first")

    @staticmethod
    def get_params(params_dict: Dict) -> str:
        result = ""
        for key, value in params_dict.items():
            if value and key[0] != '_':  # Removes the underscore used for comments
                result += f" --{key}{f' {value}' if not isinstance(value, bool) else ''}"
        return result

    def parse_input(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("-I", '--iterations', nargs=1, type=int, default=[1],
                            help='(int) iterations to be done')
        parser.add_argument("-CF", '--config_file', nargs=1, type=str, required=True)
        return parser.parse_args()

    def __call__(self, *args, **kwargs):
        args = self.parse_input()

        with open(args.config_file[0]) as json_file:
            config_json = json.load(json_file)
            self.name = str(json_file).split('_')[0]

        for _ in range(args.iterations[0]):
            for param in config_json:
                # Launch the batch jobs
                param_str = self.get_params(param)
                try:
                    self.submit_job(self.run_simulation(param_str))
                except:
                    print(f"Error with a job. Running next one.")


class LocalRunner(SimulationScheduler):

    def run_simulation(self, params: str):
        return f"""python3 -o principal_simulation.py{params}"""

    @staticmethod
    def submit_job(job):
        with open('job.sh', 'w') as fp:
            fp.write(job)
        os.system("chmod +x job.sh")
        os.system("./job.sh")


if __name__ == "__main__":
    LocalRunner()()
