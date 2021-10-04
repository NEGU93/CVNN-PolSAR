import os
from pdb import set_trace
from pathlib import Path
from collections import defaultdict
from typing import Dict
from cvnn.data_analysis import MonteCarloAnalyzer


def get_dictionary(root_dir: str = "/media/barrachina/data/results/Bretigny/ICASSP_2022_trials") -> Dict[str, str]:
    child_dirs = os.walk(root_dir)
    monte_dict = defaultdict(list)
    for child_dir in child_dirs:
        if "run-" in child_dir[0].split('/')[-1]:
            file_path = Path(child_dir[0]) / "model_summary.txt"
            if file_path.is_file():
                with open(file_path) as txt_sum_file:
                    simu_params = txt_sum_file.readline()
                    if (Path(child_dir[0]) / 'history_dict').is_file():
                        model_name = f"{'real' if 'real_mode' in simu_params else 'complex'}_{'coh' if 'coherency' in simu_params else 'k'}"
                        monte_dict[model_name].append(str(Path(child_dir[0]) / 'history_dict'))
                    else:
                        print("No history_dict found on path " + child_dir[0])
            else:
                print("No model_summary.txt found in " + child_dir[0])
    return monte_dict


if __name__ == "__main__":
    monte_dict = get_dictionary()
    del monte_dict['real_coh']
    del monte_dict['complex_coh']
    # hist_dict = {
    #     "k": [
    #         "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/22Wednesday/run-20h39m25/history_dict",
    #         "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/22Wednesday/run-13h40m03/history_dict"
    #     ],
    #     "coh": [
    #         "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/23Thursday/run-17h39m37/history_dict",
    #         "/mnt/point_de_montage/onera/PolSar/Bretigny-ONERA/log/2021/09September/23Thursday/run-10h44m59/history_dict"
    #     ]
    # }
    monte_analyzer = MonteCarloAnalyzer(history_dictionary=monte_dict)
    monte_analyzer.do_all()
