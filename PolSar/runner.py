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
        root_path = pathlib.Path(pathlib.Path(__file__).parent.resolve())
        self.iterations = 10
        self.default_config_path = str(root_path / "ober_simulations.json")

    @staticmethod
    def run_simulation(params: str):
        raise NotImplementedError(f"Need to implement this function first")

    @staticmethod
    def submit_job(job):
        raise NotImplementedError(f"Need to implement this function first")

    @staticmethod
    def get_params(params_dict: Dict) -> str:
        result = ""
        for key, value in params_dict.items():
            if value and key[0] != '_':     # Second if removes the underscores used for comments
                result += f" --{key}{f' {value}' if not isinstance(value, bool) else ''}"
        return result

    def parse_input(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--iterations', nargs=1, type=int, default=[self.iterations],
                            help='(int) iterations to be done')
        parser.add_argument('--config_file', nargs=1, type=str, default=[self.default_config_path])
        return parser.parse_args()

    def __call__(self, *args, **kwargs):
        args = self.parse_input()

        with open(args.config_file[0]) as json_file:
            config_json = json.load(json_file)
        config_json = [add_constants(conf) for conf in config_json]        # TODO: Horrible, think this better.
        config_json = [item for sublist in config_json for item in sublist]

        for iterations in range(args.iterations[0]):
            for param in config_json:
                # Launch the batch jobs
                param_str = self.get_params(param)
                try:
                    self.submit_job(self.run_simulation(param_str))
                except:
                    print(f"Error with a job. Running next one.")


class LocalRunner(SimulationScheduler):

    @staticmethod
    def run_simulation(params: str):
        return f"python3 principal_simulation.py{params}"

    @staticmethod
    def submit_job(job):
        with open('job.sh', 'w') as fp:
            fp.write(job)
        os.system("chmod +x job.sh")
        os.system("./job.sh")


if __name__ == "__main__":
    LocalRunner()()

