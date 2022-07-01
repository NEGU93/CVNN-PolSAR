import os
import json
import re
from numpy import linspace
import numpy as np
from math import sqrt
from pdb import set_trace
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from cvnn.utils import REAL_CAST_MODES
from principal_simulation import get_final_model_results, _get_dataset_handler
from typing import List, Optional, Union
from dataset_reader import COLORS

AVAILABLE_LIBRARIES = set()
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import plotly.express as px

    AVAILABLE_LIBRARIES.add('plotly')
except ImportError as e:
    print("Plotly not installed, consider installing it to get more plotting capabilities")
try:
    import matplotlib.pyplot as plt

    AVAILABLE_LIBRARIES.add('matplotlib')
    DEFAULT_MATPLOTLIB_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
    try:
        import seaborn as sns

        DEFAULT_MATPLOTLIB_COLORS = sns.color_palette()  # plt.rcParams['axes.prop_cycle'].by_key()['color']
        AVAILABLE_LIBRARIES.add('seaborn')
    except ImportError as e:
        print("Seaborn not installed, consider installing it to get more plotting capabilities")
    try:
        import tikzplotlib

        AVAILABLE_LIBRARIES.add('tikzplotlib')
    except ImportError as e:
        print("Tikzplotlib not installed, consider installing it to get more plotting capabilities")

except ImportError as e:
    print("Matplotlib not installed, consider installing it to get more plotting capabilities")

DEFAULT_PLOTLY_COLORS = [
    'rgb(31, 119, 180)',  # Blue
    'rgb(255, 127, 14)',  # Orange
    'rgb(44, 160, 44)',  # Green
    'rgb(214, 39, 40)',
    'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
    'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
    'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
]


@dataclass
class Resolution:
    width: int
    height: int


RESOLUTIONS_16_9 = {
    'lowest': Resolution(1024, 576),
    'low': Resolution(1152, 648),
    'HD': Resolution(1280, 720),  # 720p
    'FHD': Resolution(1920, 1080),  # 1080p
    'QHD': Resolution(2560, 1440),  # 1440p
    'UHD': Resolution(2560, 1440)  # 4K or 2160p
}
RESOLUTIONS_4_3 = {
    '640×480': Resolution(640, 480),
    '800×600': Resolution(800, 600),
    '960×720': Resolution(960, 720),
    '1024×768': Resolution(1024, 768),
    '1280×960': Resolution(1280, 960),
    # https://www.comtech-networking.com/blog/item/4-what-is-the-screen-resolution-or-the-aspect-ratio-what-do-720p-1080i-1080p-mean/
}

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True,
    'toImageButtonOptions': {
        'format': 'svg',  # one of png, svg, jpeg, webp
        # 'filename': 'custom_image',
        'height': RESOLUTIONS_4_3['800×600'].height,
        'width': RESOLUTIONS_4_3['800×600'].width,
        'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
    }
}


def add_transparency(color='rgb(31, 119, 180)', alpha=0.5):
    pattern = re.compile("^rgb\([0-9]+, [0-9]+, [0-9]+\)$")
    if not re.match(pattern, color):
        raise ValueError(f"Unrecognized color format ({color})")
    color = re.sub("^rgb", "rgba", color)
    color = re.sub("\)$", ", {})".format(alpha), color)
    return color


class ResultReader:

    def _search_for_root_file(self):
        if os.path.exists("/media/barrachina/data/results/new method"):
            return "/media/barrachina/data/results/new method"
        elif os.path.exists("D:/results/new method"):
            return "D:/results/new method"
        else:
            raise FileNotFoundError("No path was given to ResultReader and it could not be automatically detected.")

    def __init__(self, root_dir: str = None):
        """
        Finds all paths in a given `root_dir` directory
        :param root_dir:
        """
        if root_dir is None:
            root_dir = self._search_for_root_file()
        child_dirs = os.walk(root_dir)
        monte_dict = defaultdict(lambda: defaultdict(list))
        for child_dir in child_dirs:
            # set_trace()
            if "run-" in os.path.split(child_dir[0])[-1]:
                file_path = Path(child_dir[0]) / "model_summary.txt"
                if file_path.is_file():
                    with open(file_path) as txt_sum_file:
                        simu_params = txt_sum_file.readline()
                        if (Path(child_dir[0]) / 'history_dict.csv').is_file():
                            # TODO: Verify model and other strings are valid
                            params = {
                                "dataset": self._get_dataset(simu_params), "model": self._get_model(simu_params),
                                "dtype": self._get_real_mode(simu_params),
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset_method": self._get_dataset_method(simu_params),
                                "balance": self._get_balance(simu_params),
                                "equiv_technique": self._get_equiv_technique(simu_params)
                            }
                            monte_dict[json.dumps(params, sort_keys=True)]["image"].append(
                                str(Path(child_dir[0]) / 'prediction.png'))
                            if not os.path.isfile(monte_dict[json.dumps(params, sort_keys=True)]["image"][-1]):
                                print(f"Generating picture predicted figure on {file_path}.\n"
                                      "This will take a while but will only be done once in a lifetime")
                                # If I dont have the image I generate it
                                dataset_name = params["dataset"].upper()
                                if dataset_name == "BRETIGNY":  # For version compatibility
                                    dataset_name = "BRET"
                                mode = "t" if 'coherency' in simu_params else "s"
                                dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode,
                                                                       balance=(params['balance'] == "dataset"))
                                get_final_model_results(Path(child_dir[0]), dataset_handler=dataset_handler,
                                                        use_mask=False,
                                                        model_name=params["model"],
                                                        equiv_technique=params["equiv_technique"],
                                                        tensorflow='tensorflow' in simu_params,
                                                        complex_mode='real_mode' not in simu_params,
                                                        channels=6 if 'coherency' in simu_params else 3,
                                                        dropout={
                                                            "downsampling": None,
                                                            "bottle_neck": None,
                                                            "upsampling": None
                                                        })
                            if (Path(child_dir[0]) / 'evaluate.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["eval"].append(
                                    str(Path(child_dir[0]) / 'evaluate.csv'))
                            if (Path(child_dir[0]) / 'train_confusion_matrix.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["train_conf"].append(
                                    str(Path(child_dir[0]) / 'train_confusion_matrix.csv'))
                            if (Path(child_dir[0]) / 'val_confusion_matrix.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["val_conf"].append(
                                    str(Path(child_dir[0]) / 'val_confusion_matrix.csv'))
                            if (Path(child_dir[0]) / 'test_confusion_matrix.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["test_conf"].append(
                                    str(Path(child_dir[0]) / 'test_confusion_matrix.csv'))
                            monte_dict[json.dumps(params, sort_keys=True)]["data"].append(
                                str(Path(child_dir[0]) / 'history_dict.csv'))
                        else:
                            print("No history_dict found on path " + child_dir[0])
                else:
                    print("No model_summary.txt found in " + child_dir[0])
        self.monte_dict = monte_dict
        self.df = pd.DataFrame()
        for params in self.monte_dict.keys():
            self.df = pd.concat([self.df, pd.DataFrame(json.loads(params), index=[0])], ignore_index=True)
        self.df.sort_values(
            by=['dataset', 'model', 'dtype', 'library', 'dataset_mode', 'dataset_method', 'balance']).reset_index(
            drop=True)

    # Getters

    def get_image(self, json_key: str):
        return self.monte_dict[json_key]['image']

    def get_data(self, json_key: str):
        return self.monte_dict[json_key]['data']

    def data_exists(self, json_key: str):
        return bool(self.monte_dict[json_key]['data'])

    def get_stats(self, json_key: str):
        """
        The 'stats' key is lazy. It will only be calculated on the first call to this function
        """
        if len(self.monte_dict[json_key]['stats']) == 0:
            pandas_dict = self.get_pandas_data(json_key=json_key)
            self.monte_dict[json_key]['stats'] = pandas_dict.groupby('epoch').describe()
        return self.monte_dict[json_key]['stats']

    def get_pandas_data(self, json_key: str):
        """
        The 'stats' key is lazy. It will only be calculated on the first call to this function
        """
        pandas_dict = pd.DataFrame()
        for data_results_dict in self.monte_dict[json_key]['data']:
            result_pandas = pd.read_csv(data_results_dict, index_col=False)
            pandas_dict = pd.concat([pandas_dict, result_pandas], sort=False)
        return pandas_dict

    def get_eval_stats(self, json_key: str):
        if len(self.monte_dict[json_key]['eval_stats']) == 0:
            pandas_dict = self.get_final_results_eval(json_key)
            self.monte_dict[json_key]['eval_stats'] = pandas_dict.groupby(pandas_dict.index).describe()
        return self.monte_dict[json_key]['eval_stats']

    def get_final_results_eval(self, json_key: str):
        if len(self.monte_dict[json_key]['eval_results']) == 0:
            pandas_dict = pd.DataFrame()
            for data_results_dict in self.monte_dict[json_key]['eval']:
                result_pandas = pd.read_csv(data_results_dict, index_col=0)
                pandas_dict = pd.concat([pandas_dict, result_pandas], sort=False)
            self.monte_dict[json_key]['eval_results'] = pandas_dict
        return self.monte_dict[json_key]['eval_results']

    def get_eval_data(self, json_key: str):
        # TODO: I think eval_data and eval_results are the same XD
        if len(self.monte_dict[json_key]['eval_data']) == 0:
            self.monte_dict[json_key]['eval_data'] = []
            for data_results_dict in self.monte_dict[json_key]['eval']:
                result_pandas = pd.read_csv(data_results_dict, index_col=0)
                self.monte_dict[json_key]['eval_data'].append(result_pandas)
        return self.monte_dict[json_key]['eval_data']

    def get_conf_stats(self, json_key: str):
        if len(self.monte_dict[json_key]['conf_stats']) == 0:
            cm = []
            # TODO: Put separated function. Repeat code twice
            if len(self.monte_dict[json_key]['train_conf']) == 0:
                raise ValueError(f"No simulations results found for json_key:\n{json_key}")
            for path in self.monte_dict[json_key]['train_conf']:
                tmp_cm = pd.read_csv(path, index_col=0)
                tmp_cm = (tmp_cm.astype('float').T / tmp_cm.drop('Total', axis=1).sum(axis=1)).T
                cm.append(tmp_cm)
            cm_concat = pd.concat(tuple(cm))
            cm_group = cm_concat.groupby(cm_concat.index)
            self.monte_dict[json_key]['conf_stats'].append(cm_group.mean())
            cm = []
            for path in self.monte_dict[json_key]['val_conf']:
                tmp_cm = pd.read_csv(path, index_col=0)
                tmp_cm = (tmp_cm.astype('float').T / tmp_cm.drop('Total', axis=1).sum(axis=1)).T
                cm.append(tmp_cm)
            cm_concat = pd.concat(tuple(cm))
            cm_group = cm_concat.groupby(cm_concat.index)
            self.monte_dict[json_key]['conf_stats'].append(cm_group.mean())
            if len(self.monte_dict[json_key]['test_conf']) != 0:
                for path in self.monte_dict[json_key]['test_conf']:
                    tmp_cm = pd.read_csv(path, index_col=0)
                    tmp_cm = (tmp_cm.astype('float').T / tmp_cm.drop('Total', axis=1).sum(axis=1)).T
                    cm.append(tmp_cm)
                cm_concat = pd.concat(tuple(cm))
                cm_group = cm_concat.groupby(cm_concat.index)
                self.monte_dict[json_key]['conf_stats'].append(cm_group.mean())
        return self.monte_dict[json_key]['conf_stats']

    def get_eval_stat_string(self, json_key, dataset, stat: str, variable: str):
        eval_stats = self.get_eval_stats(json_key=json_key)
        if dataset not in eval_stats.keys():
            return f"{dataset} key not available"
        stat = stat.lower()
        if stat == 'mean':
            return f"{eval_stats[dataset]['mean'][variable]:.2%} +- {max(eval_stats[dataset]['std'][variable] / sqrt(eval_stats[dataset]['count'][variable]), 0.0001):.2%}"
        if stat == 'median':
            return f"{eval_stats[dataset]['50%'][variable]:.2%} +- {max(1.57 * (eval_stats[dataset]['75%'][variable] - eval_stats[dataset]['25%'][variable]) / sqrt(eval_stats[dataset]['count'][variable]), 0.0001):.2%}"
        if stat == 'iqr':
            return f"{eval_stats[dataset]['25%'][variable]:.2%} - {eval_stats[dataset]['75%'][variable]:.2%}"
        if stat == 'range':
            return f"{eval_stats[dataset]['min'][variable]:.2%} - {eval_stats[dataset]['max'][variable]:.2%}"

    def get_total_count(self, json_key):
        return self.get_eval_stats(json_key=json_key)['train']['count'][0]

    def find_closest_to(self, json_key, dataset, key_to_find, metric):
        if key_to_find == 'median':
            key_to_find = '50%'
        data = self.get_eval_stats(json_key=json_key)
        diff = 1.
        closest_path = None
        for data_results_dict in self.monte_dict[json_key]['eval']:
            result_pandas = pd.read_csv(data_results_dict, index_col=0)
            tmp_diff = np.abs(data[dataset][key_to_find][metric] - result_pandas[dataset][metric])
            # print(result_pandas[dataset][metric])
            if tmp_diff < diff:
                closest_path = data_results_dict
                diff = tmp_diff
            if diff == 0.:
                print("Found exact value")
                return closest_path
        print("Found close value")
        return closest_path

    """
    Methods to parse simulation parameters
    """

    @staticmethod
    def _get_model(simu_params):
        try:
            model_index = simu_params.split().index('--model')
        except ValueError:
            model_index = -1
        return f"{simu_params.split()[model_index + 1] if model_index != -1 else 'cao'}".lower()

    @staticmethod
    def _get_dataset(simu_params):
        try:
            dataset_index = simu_params.split().index('--dataset')
        except ValueError:
            dataset_index = -1
        return_value = f"{simu_params.split()[dataset_index + 1] if dataset_index != -1 else 'SF-AIRSAR'}".upper()
        if return_value == "BRETIGNY":
            return_value = "BRET"
        return return_value

    @staticmethod
    def _get_equiv_technique(simu_params):
        try:
            equiv_technique_index = simu_params.split().index('--equiv_technique')
        except ValueError:
            equiv_technique_index = -1
        return f"{simu_params.split()[equiv_technique_index + 1] if equiv_technique_index != -1 else 'ratio_tp'}".lower()

    @staticmethod
    def _get_balance(simu_params):
        try:
            balance_index = simu_params.split().index('--balance')
            value = simu_params.split()[balance_index + 1].lower()
            if value in {'loss', 'dataset'}:
                return value
            else:
                return 'none'
        except ValueError:
            return 'none'

    @staticmethod
    def _get_dataset_method(simu_params):
        try:
            dataset_method_index = simu_params.split().index('--dataset_method')
        except ValueError:
            dataset_method_index = -1
        return f"{simu_params.split()[dataset_method_index + 1] if dataset_method_index != -1 else 'random'}"

    @staticmethod
    def _get_real_mode(simu_params):
        if 'real_mode' in simu_params:
            real_mode_index = simu_params.split().index('--real_mode')
            next_value = simu_params.split()[real_mode_index + 1] if real_mode_index + 1 < len(simu_params.split()) \
                else 'real_imag'
            return next_value if next_value in REAL_CAST_MODES else 'real_imag'
        elif 'tensorflow' in simu_params:
            return 'real_imag'
        else:
            return 'complex'


"""
    MonteCarlo Plotter
"""


class MonteCarloPlotter:

    def plot(self, data, keys: List[str], ax=None):
        """
        :param data:
        :param ax: (Optional) axis on which to plot the data
        :param keys:
        :return:
        """
        self._plot_line_confidence_interval_matplotlib(ax=ax, keys=keys, stats=data)

    def _plot_line_confidence_interval_matplotlib(self, keys: List[str], stats, ax=None, showfig=False, x_axis='epoch'):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        for i, key in enumerate(keys):
            x = stats.index.values.tolist()
            data_mean = stats[key]['mean'].tolist()
            data_max = stats[key]['max'].tolist()
            data_min = stats[key]['min'].tolist()
            data_50 = stats[key]['50%'].tolist()
            data_25 = stats[key]['25%'].tolist()
            data_75 = stats[key]['75%'].tolist()
            ax.plot(x, data_mean, color=DEFAULT_MATPLOTLIB_COLORS[i % len(DEFAULT_MATPLOTLIB_COLORS)],
                    label=key)
            ax.plot(x, data_50, '--', color=DEFAULT_MATPLOTLIB_COLORS[i % len(DEFAULT_MATPLOTLIB_COLORS)])
            # label=key + ' median')
            ax.fill_between(x, data_25, data_75, color=DEFAULT_MATPLOTLIB_COLORS[i % len(DEFAULT_MATPLOTLIB_COLORS)],
                            alpha=.4)  # , label=key + ' interquartile')
            ax.fill_between(x, data_min, data_max, color=DEFAULT_MATPLOTLIB_COLORS[i % len(DEFAULT_MATPLOTLIB_COLORS)],
                            alpha=.15)  # , label=key + ' border')
        # title = title[:-3] + key
        #
        # ax.set_title(title)
        ax.set_xlabel(x_axis)
        # ax.set_ylabel(key)
        ax.grid()
        ax.legend()
        # set_trace()
        if showfig and fig:
            fig.show()


class SeveralMonteCarloPlotter:

    @staticmethod
    def _save_show_figure(fig, showfig=False, savefile=None, extension=".svg"):
        if savefile is not None:
            if not savefile.endswith(extension):
                savefile += extension
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            fig.savefig(savefile, transparent=True)
            if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
                raise ModuleNotFoundError(
                    "No Tikzplotlib installed, function will not save tikz file")
            else:
                tikzplotlib.save(Path(str(savefile).split('.')[0] + ".tikz"))
        if showfig and fig:
            fig.show()
        plt.close(fig)

    @staticmethod
    def _to_string_label(ylabel):
        return ylabel.replace("_", " ").replace("val", "validation")

    @staticmethod
    def get_metric_and_dataset_from_key(key: str):
        """
        Testing:
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("val_average_accuracy") == ("val", "average_accuracy")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("val_average") == ("val", "average")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("average_accuracy") == ("train", "average_accuracy")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("average_accuracy") == ("train", "average_accuracy")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("test_average_accuracy") == ("test", "average_accuracy")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("test_accuracy") == ("test", "accuracy")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("val_losodan") == ("val", "losodan")
        assert SeveralMonteCarloPlotter.get_metric_and_dataset_from_key("losodan") == ("train", "losodan")
        """
        split = key.split('_')
        if len(split) == 1 or split[0] not in ["val", "test"]:
            return 'train', key
        else:
            return tuple([split[0], "_".join(split[1:])])

    @staticmethod
    def get_key_from_metric_and_dataset(dataset: str, metric: str):
        metric = metric.replace(" ", "_")
        if dataset == "train":
            return metric
        else:
            return '_'.join([dataset, metric])

    """--------------
        BAR PLOT
    --------------"""
    def per_class_bar_plot(self, labels: List[str], data: List,
                           dataset,
                           library="seaborn", print_values=True,
                           showfig=False, savefile=None, colors=None, extension: str = ".svg", min_lim=0.0):
        """
        Plots bar accuracy per class of each simulation model separately.
        :param labels:
        :param data:
        :param index:
        :param library:
        :param showfig:
        :param savefile:
        :param colors:
        :param extension:
        :param min_lim:
        :return:
        """
        if library == "seaborn":
            self._matplotlib_bar_plot(labels=labels, data=data, dataset=dataset,
                                      showfig=showfig, savefile=savefile, print_values=print_values,
                                      colors=colors, extension=extension, min_lim=min_lim)
        elif library == "plotly":
            self._plotly_bar_plot(labels=labels, data=data, dataset=dataset,
                                  showfig=showfig, savefile=savefile, colors=colors)

    def _plotly_bar_plot(self, labels: List[str], data: List, dataset, showfig=False, savefile=None, colors=None):
        d_index = ['train', 'val', 'test'].index(dataset)
        savefig = False
        if savefile is not None:
            savefig = True
        df_list = []
        for dat in data:
            tmp_dict = {f"class {j}": dat[d_index][str(j)][str(j)] for j in range(len(dat[d_index])-1)}
            df_list.append(tmp_dict)
        plotly_colors = [plotly.colors.label_rgb(color) for color in [plotly.colors.convert_to_RGB_255(color) for color in colors]]
        fig = go.Figure(
            data=[go.Bar(name=key, x=labels, y=[elem[key] for elem in df_list],
                         marker_color=plotly_colors[i]) for i, key in enumerate(df_list[0].keys())]
        )
        fig.update_layout(
            title=f"{' vs '.join(labels)} {dataset} per class accuracy",
            yaxis=dict(
                title=f"{dataset} accuracy",
                autorange=True,
                showgrid=True,
                dtick=0.05,
            ),
            xaxis=dict(
                title="model / classes"
            ),
            showlegend=True
        )
        if savefig:
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            if not str(savefile).endswith(".html"):
                savefile = str(savefile) + ".html"
            plotly.offline.plot(fig,
                                filename=str(savefile),
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / ("plots/lines/montecarlo_" + key.replace(" ", "_"))) + extension)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def _matplotlib_bar_plot(self, labels: List[str], data: List, dataset,
                             showfig=False, savefile=None, colors=None,
                             extension: str = ".svg", min_lim=0.0, print_values=True):
        if colors is None:
            colors = DEFAULT_MATPLOTLIB_COLORS
        borders = 0.2
        d_index = ['train', 'val', 'test'].index(dataset)
        classes = len(data[0][d_index]) - 1  # 3
        x = range(len(labels))  # 4
        offset = linspace(-borders, borders, classes)  # classes = 3
        fig, ax = plt.subplots()
        classes_results = []
        for i, mc_run in enumerate(data):
            tmp = []
            for j in range(len(mc_run[d_index]) - 1):
                tmp.append(mc_run[d_index][str(j)][str(j)])
            classes_results.append(tmp)
        arr = np.array(classes_results)
        for j in range(len(offset)):
            ax.bar(x + offset[j], arr[:, j], align='center', width=3 * borders / classes,
                   tick_label=labels, color=colors[j])
            if print_values:
                for location, val in enumerate(arr[:, j]):
                    ax.text((x + offset[j] - 0.05)[location], val, f"{val:.2%}",
                            color='black', fontweight='bold', rotation=90, ha='left', va='top')
        max_lim = 1.
        ax.set_ylim((min_lim, max_lim))
        minor_ticks = np.arange(min_lim, max_lim, 0.05)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_ylabel(f"{dataset} accuracy")
        ax.grid(axis='y', which='both')
        self._save_show_figure(fig, showfig=showfig, savefile=savefile, extension=extension)

    """--------------
        BOX PLOT
    --------------"""
    def violin_plot(self, labels: List[str], data: List,
                    key='val_accuracy', library='seaborn', showfig=False, savefile: Optional[str] = None):
        self.box_plot(labels=labels, data=data, key=key, library=library, showfig=showfig, savefile=savefile,
                      violin_plot=True)

    def box_plot(self, labels: List[str], data: List, violin_plot=False,
                 key='val_accuracy', library='seaborn', showfig=False, savefile: Optional[str] = None):
        """
        Saves/shows a box plot of the results.
        :param labels: List of labels of each simulation to compare
        :param data: List of pandas describe() dataframes wrt epochs
        :param key: String stating what to plot using tf.keras.History labels. ex. `val_accuracy` for the validation acc
        :param library: string stating the library to be used to generate the box plot.
            - `plotly <https://plotly.com/python/>`_
            - `seaborn <https://seaborn.pydata.org/>`_
        :param epoch: Which epoch to use for the box plot. If -1 (default) it will use the last epoch.
        :param showfig: If True, it will show the grated box plot
        :param savefile: String with the path + filename where to save the boxplot. If None (default) no figure is saved
        """
        if library == 'plotly':
            self._box_plot_plotly(labels=labels, mc_runs=data, key=key, showfig=showfig, savefile=savefile,
                                  violin_plot=violin_plot)
        # TODO: https://seaborn.pydata.org/examples/grouped_boxplot.html
        elif library == 'seaborn':
            self._box_plot_seaborn(labels=labels, data=data, key=key, showfig=showfig, savefile=savefile,
                                   violin_plot=violin_plot)
        else:
            raise ModuleNotFoundError(f"Library {library} requested for plotting unknown")
        return None

    def _box_plot_plotly(self, labels: List[str], mc_runs: List,
                         key='accuracy', showfig=False, savefile=None, violin_plot=False):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(f"No Plotly installed, function {self._box_plot_plotly.__name__} "
                                      f"was called but will be omitted")
        savefig = False
        if savefile is not None:
            savefig = True
        dataset, metric = self.get_metric_and_dataset_from_key(key)
        # Prepare data
        fig = go.Figure()
        # color_pal = []
        for i, mc_run in enumerate(mc_runs):
            # color_pal += sns.color_palette()[:len(df.network.unique())]
            data = mc_run[mc_run.index.isin([metric])]
            if not violin_plot:
                fig.add_trace(go.Box(
                    y=data[dataset],
                    name=labels[i],
                    whiskerwidth=0.2,
                    notched=True,  # confidence intervals for the median
                    fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.5),
                    boxpoints='suspectedoutliers',  # to mark the suspected outliers
                    line=dict(color=DEFAULT_PLOTLY_COLORS[i]),
                    boxmean=True  # Interesting how sometimes it falls outside the box
                ))
            else:
                fig.add_trace(go.Violin(
                    y=data[dataset],
                    name=labels[i],
                    fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.5),
                    line=dict(color=DEFAULT_PLOTLY_COLORS[i]),
                    box_visible=True, meanline_visible=True
                ))
        fig.update_layout(
            title=f"{' vs '.join(labels)} {dataset} {metric.replace('_', ' ')} {'violin' if violin_plot else 'box'} plot",
            yaxis=dict(
                title=f"{dataset} {metric.replace('_', ' ')}",
                autorange=True,
                showgrid=True,
                dtick=0.05,
            ),
            xaxis=dict(
              title="model"
            ),
            showlegend=True
        )
        if savefig:
            if not savefile.endswith('.html'):
                savefile += '.html'
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            plotly.offline.plot(fig, filename=savefile, config=PLOTLY_CONFIG, auto_open=showfig)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def _box_plot_seaborn(self, labels: List[str], data: List,
                          key='accuracy', showfig=False, savefile=None, extension=".svg", violin_plot=False):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(
                "No Seaborn installed, function " + self._box_plot_seaborn.__name__ + " was called but will be omitted")
        dataset, metric = self.get_metric_and_dataset_from_key(key)
        # Prepare data
        frames = []
        # color_pal = []
        assert len(data) == len(labels), f"len(data) = {len(data)} does not match len(labels) = {len(labels)}"
        for i, mc_run in enumerate(data):
            t_data = mc_run[mc_run.index.isin([metric])].assign(name=labels[i])
            frames.append(t_data)
        result = pd.concat(frames)
        # Run figure
        fig = plt.figure()
        if not violin_plot:
            ax = sns.boxplot(x='name', y=dataset, data=result, boxprops=dict(alpha=.3),
                             palette=sns.color_palette(), notch=True)
        else:
            ax = sns.violinplot(x='name', y=dataset, data=result,   # boxprops=dict(alpha=.3),
                                palette=sns.color_palette())
        ax.grid(axis='y', which='major')
        ax.set_ylabel(f"{dataset} {metric.replace('_', ' ')}")
        ax.set_xlabel("model")
        # palette=color_pal)
        # sns.despine(offset=1, trim=True)
        # Make black lines the color of the box
        for i, artist in enumerate(ax.artists):
            col = artist.get_facecolor()[:-1]  # the -1 removes the transparency
            artist.set_edgecolor(col)
            for j in range(i * 6, i * 6 + 6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        self._save_show_figure(fig, showfig=showfig, savefile=savefile, extension=extension)
        return fig, ax

    """--------------
        LINE PLOT
    --------------"""
    def plot(self, data, labels: List[str], keys: Union[str, List[str]] = "val_accuracy",
             ax=None, library="seaborn", showfig=False, savefile=None):
        """
        :param data:
        :param ax: (Optional) axis on which to plot the data
        :param keys:
        :return:
        """
        if library == "seaborn":
            self._plot_line_confidence_interval_matplotlib(ax=ax, keys=keys, data=data, labels=labels, showfig=showfig,
                                                           savefile=savefile)
        elif library == "plotly":
            self._plot_line_confidence_interval_plotly(keys=keys, labels=labels, data=data, showfig=showfig,
                                                       savefile=savefile)

    def _plot_line_confidence_interval_matplotlib(self, keys: Union[List[str], str], labels: List[str],
                                                  data, ax=None, showfig=False, x_axis='epoch',
                                                  savefile=None, extension=".svg"):
        if isinstance(keys, str):
            keys = [keys]
        keys = list(keys)
        metrics = set([self.get_metric_and_dataset_from_key(key)[1] for key in keys])
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        for i, dat in enumerate(data):
            for j, key in enumerate(keys):
                stats = dat.groupby("epoch").describe()
                x = stats.index.values.tolist()
                data_mean = stats[key]['mean'].tolist()
                data_max = stats[key]['max'].tolist()
                data_min = stats[key]['min'].tolist()
                data_50 = stats[key]['50%'].tolist()
                data_25 = stats[key]['25%'].tolist()
                data_75 = stats[key]['75%'].tolist()
                ax.plot(x, data_mean,
                        color=DEFAULT_MATPLOTLIB_COLORS[(i * len(keys) + j) % len(DEFAULT_MATPLOTLIB_COLORS)],
                        label=labels[i] + " " + key.replace('_', ' '))
                ax.plot(x, data_50, '--',
                        color=DEFAULT_MATPLOTLIB_COLORS[(i * len(keys) + j) % len(DEFAULT_MATPLOTLIB_COLORS)])
                # label=key + ' median')
                ax.fill_between(x, data_25, data_75,
                                color=DEFAULT_MATPLOTLIB_COLORS[(i * len(keys) + j) % len(DEFAULT_MATPLOTLIB_COLORS)],
                                alpha=.4)  # , label=key + ' interquartile')
                ax.fill_between(x, data_min, data_max,
                                color=DEFAULT_MATPLOTLIB_COLORS[(i * len(keys) + j) % len(DEFAULT_MATPLOTLIB_COLORS)],
                                alpha=.15)  # , label=key + ' border')
        # title = title[:-3] + key
        #
        # ax.set_title(title)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("; ".join(keys).replace('_', ' '))
        ax.grid()
        ax.legend()
        self._save_show_figure(fig, showfig=showfig, savefile=savefile, extension=extension)

    def _plot_line_confidence_interval_plotly(self, keys: Union[List[str], str], labels: List[str],
                                              data, ax=None, showfig=False, x_axis='epoch',
                                              savefile=None, full_border=True):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(f"No Plotly installed, function {self._box_plot_plotly.__name__} "
                                      f"was called but will be omitted")
        savefig = False
        if savefile is not None:
            savefig = True
        if isinstance(keys, str):
            keys = [keys]
        fig = go.Figure()
        title = ""
        for i, (dat, label) in enumerate(zip(data, labels)):
            for key in keys:
                x = dat[x_axis].unique().tolist()
                x_rev = x[::-1]
                stats = dat.groupby("epoch").describe()
                data_mean = stats[key]['mean'].tolist()
                data_max = stats[key]['max'].tolist()
                data_min = stats[key]['min'][::-1].tolist()
                data_50 = stats[key]['50%'].tolist()
                data_25 = stats[key]['25%'][::-1].tolist()
                data_75 = stats[key]['75%'].tolist()
                if full_border:
                    fig.add_trace(go.Scatter(
                        x=x + x_rev,
                        y=data_max + data_min,
                        fill='toself',
                        fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)], 0.1),
                        line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)], 0),
                        showlegend=True,
                        name=label.replace('_', ' ') + " borders",
                    ))
                fig.add_trace(go.Scatter(
                    x=x + x_rev,
                    y=data_75 + data_25,
                    fill='toself',
                    fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)], 0.2),
                    line_color=add_transparency(DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)], 0),
                    showlegend=True,
                    name=label.replace('_', ' ') + " interquartile",
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=data_mean,
                    line_color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)],
                    name=label.replace('_', ' ') + " mean",
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=data_50,
                    line=dict(color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)], dash='dash'),
                    name=label.replace('_', ' ') + " median",
                ))
        for label in labels:
            title += label.replace('_', ' ') + ' vs '
        title = title[:-3]
        for key in keys:
            title += self._to_string_label(key) + " "

        fig.update_traces(mode='lines')
        fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=self._to_string_label(key))

        if savefig:
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            plotly.offline.plot(fig,
                                filename=str(savefile) + ".html",
                                config=PLOTLY_CONFIG, auto_open=showfig)
            # fig.write_image(str(self.path / ("plots/lines/montecarlo_" + key.replace(" ", "_"))) + extension)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    """--------------------
        HISTOGRAM PLOT
    --------------------"""
    def histogram_plot(self, labels: List[str], data: List, key='val_accuracy', library='seaborn',
                       showfig=False, savefile: Optional[str] = None, extension=".svg"):
        if library == 'plotly':
            self._plotly_histogram_plot(labels=labels, mc_runs=data, key=key, showfig=showfig, savefile=savefile)
        elif library == 'seaborn':
            self._matplotlib_histogram_plot(labels=labels, data=data, key=key,
                                            showfig=showfig, savefile=savefile, extension=extension)
        else:
            raise ModuleNotFoundError(f"Library {library} requested for plotting unknown")
        return None

    def _plotly_histogram_plot(self, labels: List[str], mc_runs: List,
                         key='accuracy', showfig=False, savefile=None):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(f"No Plotly installed, function {self._box_plot_plotly.__name__} "
                                      f"was called but will be omitted")
        savefig = False
        if savefile is not None:
            savefig = True
        dataset, metric = self.get_metric_and_dataset_from_key(key)
        fig = go.Figure()
        for i, mc_run in enumerate(mc_runs):
            # color_pal += sns.color_palette()[:len(df.network.unique())]
            data = mc_run[mc_run.index.isin([metric])]
            fig.add_trace(go.Histogram(
                x=data[dataset],
                name=labels[i],
                # line=dict(color=DEFAULT_PLOTLY_COLORS[i]),
            ))
        fig.update_layout(
            # barmode='overlay',
            title=f"{' vs '.join(labels)} {dataset} {metric.replace('_', ' ')} histogram",
            yaxis=dict(
                title="count",
            ),
            xaxis=dict(
                title=f"{dataset} {metric.replace('_', ' ')}",
            ),
            showlegend=True
        )
        # fig.update_traces(opacity=0.75)
        if savefig:
            if not savefile.endswith('.html'):
                savefile += '.html'
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            plotly.offline.plot(fig, filename=savefile, config=PLOTLY_CONFIG, auto_open=showfig)
        elif showfig:
            fig.show(config=PLOTLY_CONFIG)

    def _matplotlib_histogram_plot(self, labels: List[str], data: List, key='accuracy',
                                   showfig=False, savefile=None, extension=".svg"):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(
                "No Seaborn installed, function " + self._box_plot_seaborn.__name__ + " was called but will be omitted")
        dataset, metric = self.get_metric_and_dataset_from_key(key)
        # Prepare data
        frames = []
        # color_pal = []
        assert len(data) == len(labels), f"len(data) = {len(data)} does not match len(labels) = {len(labels)}"
        for i, mc_run in enumerate(data):
            t_data = mc_run[mc_run.index.isin([metric])].assign(name=labels[i])
            frames.append(t_data)
        result = pd.concat(frames)
        # Run figure
        fig = plt.figure()
        ax = sns.histplot(x=dataset, data=result, hue="name", kde=False)
        ax.set_xlabel(f"{dataset} {metric.replace('_', ' ')}")
        ax.set_ylabel("count")
        self._save_show_figure(fig, showfig=showfig, savefile=savefile, extension=extension)
        return fig, ax

    """
        CONFUSION MATRIX (HEATMAP)
    """
    def confusion_matrix(self, labels: List[str], data: List, dataset: str,
                         library="plotly", print_values=True,
                         showfig=False, savefile=None, colors=None, extension: str = ".svg", min_lim=0.0):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(f"No Plotly installed, function {self._box_plot_plotly.__name__} "
                                      f"was called but will be omitted")
        savefig = False
        if savefile is not None:
            savefig = True
        if savefig:
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
        for lab, dat in zip(labels, data):
            conf_mat = dat[['val', 'train', 'test'].index(dataset)]
            z = conf_mat.values.tolist()
            annotations = [[f"{100*col:.2f}%" for col in row] for row in z]
            fig = ff.create_annotated_heatmap(z, x=conf_mat.columns.tolist(), y=conf_mat.index.tolist(),
                                              annotation_text=annotations, colorscale='Matter')
            # fig = go.Figure(data=go.Heatmap(
            #     z=z, x=conf_mat.columns.tolist(), y=conf_mat.index.tolist(),
            #     colorscale='Matter'
            # ))
            fig.update_layout(
                yaxis=dict(visible=True, autorange='reversed')
            )
            if savefig:
                tmp_savefile = str(savefile) + lab.replace(' ', '_') + ".html"
                plotly.offline.plot(fig, filename=str(tmp_savefile), config=PLOTLY_CONFIG, auto_open=showfig)
                # set_trace()
            elif showfig:
                fig.show(config=PLOTLY_CONFIG)


def plot_all(simulations, models_params, library, root_path, labels, colors=None, showfig=False,
             y_min_bar_plot=0.0, print_values_bar_plot=True):
    datasets = ['val', 'train', 'test']
    metrics = ['loss', 'accuracy', 'average_accuracy']
    data = [simulations.get_pandas_data(k) for k in models_params]
    final_results = [simulations.get_final_results_eval(k) for k in models_params]
    conf_stats = [simulations.get_conf_stats(k) for k in models_params]
    for dset in datasets:
        for metric in metrics:
            key = SeveralMonteCarloPlotter().get_key_from_metric_and_dataset(dataset=dset, metric=metric)
            if dset != "test":
                SeveralMonteCarloPlotter().plot(data=data, labels=labels, keys=key, library=library,
                                                showfig=showfig, savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/lines-plot")
            if metric == "accuracy":
                SeveralMonteCarloPlotter().confusion_matrix(labels=labels, showfig=False, library=library, dataset=dset,
                                                            data=conf_stats, print_values=print_values_bar_plot,
                                                            savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/confusion_matrix/conf_matrix_")
                SeveralMonteCarloPlotter().per_class_bar_plot(labels=labels, showfig=showfig,
                                                              data=conf_stats, colors=colors, library=library,
                                                              dataset=dset,
                                                              min_lim=y_min_bar_plot,
                                                              print_values=print_values_bar_plot,
                                                              savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/per-class-bar")
            SeveralMonteCarloPlotter().histogram_plot(labels=labels, data=final_results, showfig=showfig,
                                                      savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/histogram",
                                                      library=library, key=key)
            SeveralMonteCarloPlotter().box_plot(labels=labels, data=final_results, showfig=showfig,
                                                savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/box-plot",
                                                library=library, key=key)
            SeveralMonteCarloPlotter().violin_plot(labels=labels, data=final_results, showfig=showfig,
                                                   savefile=f"{root_path}/{f'{dset}idation' if dset == 'val' else dset}/{metric}/violin-plot",
                                                   library=library, key=key)


if __name__ == "__main__":
    PLOT_OBER = False
    PLOT_SF = False
    PLOT_FLEV = False
    PLOT_OBER_EQUIV = True
    BRET_PLOTS = False
    # simulation_results = ResultReader(root_dir="/media/barrachina/data/results/new method")
    simulation_results = ResultReader(root_dir="/media/barrachina/data/results/equiv_technique")
    # lst = list(simulation_results.monte_dict.keys())
    if PLOT_SF:
        sf_keys = [
            '{"balance": "none", "dataset": "SF-AIRSAR", "dataset_method": '
            '"random", "dataset_mode": "k", "dtype": "complex", "library": '
            '"cvnn", "model": "cao"}',
            '{"balance": "none", "dataset": "SF-AIRSAR", "dataset_method": "random", '
            '"dataset_mode": "coh", "dtype": "complex", "library": "cvnn", '
            '"model": "cao"}',
            '{"balance": "none", "dataset": "SF-AIRSAR", "dataset_method": '
            '"random", "dataset_mode": "k", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "cao"}',
            '{"balance": "none", "dataset": "SF-AIRSAR", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "cao"}'
        ]
        labels = ["CV Pauli", "CV Coh", "RV Pauli", "RV Coh"]
        plot_all(simulations=simulation_results, models_params=sf_keys, library="plotly",
                 root_path="/home/barrachina/Documents/cvnn_vs_rvnn_polsar_applications/public/assets/SF-AIRSAR/",
                 labels=labels, colors=COLORS['SF-AIRSAR'])
    if BRET_PLOTS:
        keys = []
        labels = []
        for dataset_mode in {"k", "coh"}:
            for dataset_method in {"random", "separate"}:
                keys.append(
                    f'{{"balance": "dataset", "dataset": "BRET", "dataset_method": "{dataset_method}", '
                    f'"dataset_mode": "{dataset_mode}", "dtype": "complex", "equiv_technique": "ratio_tp", '
                    f'"library": "cvnn", "model": "cao"}}'
                )
                keys.append(
                    f'{{"balance": "dataset", "dataset": "BRET", "dataset_method": "{dataset_method}", '
                    f'"dataset_mode": "{dataset_mode}", "dtype": "real_imag", "equiv_technique": "ratio_tp", '
                    f'"library": "tensorflow", "model": "cao"}}'
                )
                labels.append(f"CV-FCNN-{dataset_mode}-{dataset_method}")
                labels.append(f"RV-FCNN-{dataset_mode}-{dataset_method}")
        plot_all(simulations=simulation_results, models_params=keys, library="plotly",
                 root_path="/home/barrachina/Documents/cvnn_vs_rvnn_polsar_applications/public/assets/Bretigny/",
                 labels=labels, colors=COLORS['BRET'])
    if PLOT_OBER_EQUIV:
        keys = [
            '{"balance": "none", "dataset": "OBER", "dataset_method": "random", '
            '"dataset_mode": "coh", "dtype": "complex", "equiv_technique": "ratio_tp", "library": "cvnn", '
            '"model": "mlp"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "equiv_technique": "ratio_tp", "library": '
            '"tensorflow", "model": "mlp"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "equiv_technique": "np", "library": '
            '"tensorflow", "model": "mlp"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "equiv_technique": "alternate_tp", "library": '
            '"tensorflow", "model": "mlp"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "equiv_technique": "none", "library": '
            '"tensorflow", "model": "mlp"}'
        ]
        labels = ["CV-MLP", "RV-MLP-RATIO", "RV-MLP-NP", "RV-MLP-ALTERNATE", "RV-MLP-NONE"]     # "RV-MLP-ALTERNATE",
        plot_all(simulations=simulation_results, models_params=keys, library="seaborn",
                 root_path="/media/barrachina/data/results/graphs/equiv_technique_ober_mlp",
                 labels=labels, colors=COLORS['OBER'])
    if PLOT_FLEV:
        keys = [
            '{"balance": "none", "dataset": "FLEVOLAND", "dataset_method": "random", '
            '"dataset_mode": "coh", "dtype": "complex", "library": "cvnn", '
            '"model": "cao"}',
            '{"balance": "none", "dataset": "FLEVOLAND", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "cao"}',
        ]
        labels = ["CV-FCNN", "RV-FCNN"]
        library = "plotly"
        library_paths = {
            "seaborn": "/home/barrachina/gretsi_images/",
            "plotly": "/home/barrachina/Documents/cvnn_vs_rvnn_polsar_applications/public/assets/Flevoland/"
        }
        plot_all(simulations=simulation_results, models_params=keys, library=library,
                 y_min_bar_plot=0.8, print_values_bar_plot=False,
                 root_path=library_paths[library], labels=labels, colors=COLORS['FLEVOLAND'])
    if PLOT_OBER:
        keys = [
            '{"balance": "none", "dataset": "OBER", "dataset_method": "random", '
            '"dataset_mode": "coh", "dtype": "complex", "library": "cvnn", '
            '"model": "cao"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "cao"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "complex", "library": '
            '"cvnn", "model": "cnn"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "cnn"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "complex", "library": '
            '"cvnn", "model": "mlp"}',
            '{"balance": "none", "dataset": "OBER", "dataset_method": '
            '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
            '"tensorflow", "model": "mlp"}'
        ]
        labels = ["CV-FCNN", "RV-FCNN", "CV-CNN", "RV-CNN", "CV-MLP", "RV-MLP"]
        plot_all(simulations=simulation_results, models_params=keys, library="plotly",
                 root_path="/home/barrachina/Documents/cvnn_vs_rvnn_polsar_applications/public/assets/Oberpfaffenhofen/",
                 labels=labels, colors=COLORS['OBER'])
