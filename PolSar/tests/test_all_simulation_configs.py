import sys
import os
from pathlib import Path
from typing import Dict
import json
sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from runner import SimulationScheduler


class RunnerDebugger(SimulationScheduler):

    @staticmethod
    def get_params(params_dict: Dict) -> str:
        result = ""
        for key, value in params_dict.items():
            if value and key[0] != '_':  # Removes the underscore used for comments
                if key == "epochs":
                    value = 1
                result += f" --{key}{f' {value}' if not isinstance(value, bool) else ''}"
        return result

    def __call__(self, filename, *args, **kwargs):
        with open(filename) as json_file:
            config_json = json.load(json_file)
            self.name = str(json_file).split('_')[0]

        for param in config_json:
            param_str = self.get_params(param)
            self.submit_job(self.run_simulation(param_str))

    def run_simulation(self, params: str):
        return f"""python3 -O ../principal_simulation.py{params}"""

    @staticmethod
    def submit_job(job):
        with open('job.sh', 'w') as fp:
            fp.write(job)
        os.system("chmod +x job.sh")
        os.system("./job.sh")


def test_all_files():
    simulation_config_path = "/".join(os.path.abspath(__file__).split('/')[:-2] + ["simulations_configs"])
    for file in os.listdir(simulation_config_path):
        single_file_test(file)


def single_file_test(files):
    for file in files:
        if file.endswith(".json"):
            print(f"Testing {file}")
        simulation_config_path = "/".join(os.path.abspath(__file__).split('/')[:-2] + ["simulations_configs"])
        RunnerDebugger()(str(Path(simulation_config_path) / file))


if __name__ == "__main__":
    # test_all_files()
    single_file_test(["bretigny.json",
                      "bretigny_classification_balance.json",
                      "bretigny_classification_balance_separate.json"
                      ])
