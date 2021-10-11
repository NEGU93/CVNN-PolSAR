import os
import pickle
from pdb import set_trace
from pathlib import Path
from collections import defaultdict
from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

from typing import Dict

from cvnn.data_analysis import MonteCarloAnalyzer
from cvnn.utils import transform_to_real_map_function
from image_generator import open_saved_model, save_result_image_from_saved_model
from bretigny_dataset import get_bret_separated_dataset


def get_dictionary(root_dir: str = "/media/barrachina/data/results/Bretigny/ICASSP_2022_trials") -> Dict[str, str]:
    """
    Finds all simulations in a given `root_dir` directory
    :param root_dir:
    :return:
    """
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


def save_test_results(monte_dict, monte_analyzer):
    print("Getting test results")
    test_results = defaultdict(lambda: defaultdict(list))
    _, complex_val_data, complex_test_data = get_bret_separated_dataset(complex_mode=True, coherency=False,
                                                                        shuffle=False)
    real_test_data = complex_test_data.map(lambda img, labels: transform_to_real_map_function(img, labels))
    real_val_data = complex_val_data.map(lambda img, labels: transform_to_real_map_function(img, labels))
    for key, paths in monte_dict.items():
        for path in paths:
            complex_mode = 'complex' in key
            model = open_saved_model(Path(path).parents[0], complex_mode=complex_mode,
                                     tensorflow=False, dropout=None, coherency=False)
            if complex_mode:
                metrics_result = model.evaluate(complex_test_data)
                val_metrics_result = model.evaluate(complex_val_data)
            else:
                metrics_result = model.evaluate(real_test_data)
                val_metrics_result = model.evaluate(real_val_data)
            test_results[key]['test_loss'].append(metrics_result[0])
            test_results[key]['test_acc'].append(metrics_result[1])
            test_results[key]['test_avg_acc'].append(metrics_result[2])
            test_results[key]['val_loss'].append(val_metrics_result[0])
            test_results[key]['val_acc'].append(val_metrics_result[1])
            test_results[key]['val_avg_acc'].append(val_metrics_result[2])
    df = DataFrame()
    for network, metric in test_results.items():
        tmp_df = DataFrame.from_dict(metric)
        tmp_df['network'] = network
        df = df.append(tmp_df, ignore_index=True)
    df.to_csv(Path(monte_analyzer.path) / "test_results.csv")
    return None


def generate_masked_prediction(monte_dict):
    print("Generating masked prediction")
    for key, paths in monte_dict.items():
        for path in paths:
            complex_mode = 'complex' in key
            save_result_image_from_saved_model(Path(path).parents[0], complex_mode=complex_mode, use_mask=True)


def generate_csv_results(monte_dict):
    for key, paths in monte_dict.items():
        for path in paths:
            df = DataFrame.from_dict(pd.read_pickle(path))
            df.to_csv(Path(path).parents[0] / "run_data_1.csv")


def parse_data():
    monte_dict = get_dictionary()
    del monte_dict['real_coh']
    del monte_dict['complex_coh']
    monte_dict['real_k'] = monte_dict['real_k'][:-2]

    monte_analyzer = MonteCarloAnalyzer(history_dictionary=monte_dict)
    monte_analyzer.save_stat_results()
    monte_analyzer.do_all()
    # save_test_results(monte_dict, monte_analyzer)
    # generate_csv_results(monte_dict)
    # generate_masked_prediction(monte_dict)


def get_stats_of_results():
    path = "/home/barrachina/Documents/onera/PolSar/Bretigny-ONERA/log/montecarlo/2021/10October/05Tuesday/run-16h22m12/test_results.csv"
    df = DataFrame.from_dict(pd.read_csv(path))
    networks_availables = df.network.unique()
    for net in networks_availables:
        data = df[df.network == net].describe()
        data.to_csv(Path(path).parents[0] / (net + "_max_stat_result.csv"))
    fig = plt.figure()
    ax = sns.boxplot(x="network", y="val_acc", hue="network", data=df[df.network != "real_k_b"],
                     boxprops=dict(alpha=.3))
    for i, artist in enumerate(ax.artists):
        col = artist.get_facecolor()[:-1]  # the -1 removes the transparency
        artist.set_edgecolor(col)
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
    fig.savefig(Path(path).parents[0] / "tikz_box_plot.png", transparent=False)
    tikzplotlib.save(Path(path).parents[0] / "tikz_box_plot.tex")


if __name__ == "__main__":
    # get_stats_of_results()
    parse_data()




