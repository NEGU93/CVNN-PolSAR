import os
import json
import re
from pdb import set_trace
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from cvnn.utils import REAL_CAST_MODES
from principal_simulation import get_final_model_results, _get_dataset_handler
from typing import List, Optional

AVAILABLE_LIBRARIES = set()
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    AVAILABLE_LIBRARIES.add('plotly')
except ImportError as e:
    print("Plotly not installed, consider installing it to get more plotting capabilities")
try:
    import matplotlib.pyplot as plt

    AVAILABLE_LIBRARIES.add('matplotlib')
    DEFAULT_MATPLOTLIB_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
except ImportError as e:
    print("Matplotlib not installed, consider installing it to get more plotting capabilities")
if 'matplotlib' in AVAILABLE_LIBRARIES:
    try:
        import seaborn as sns

        AVAILABLE_LIBRARIES.add('seaborn')
    except ImportError as e:
        print("Seaborn not installed, consider installing it to get more plotting capabilities")
    try:
        import tikzplotlib

        AVAILABLE_LIBRARIES.add('tikzplotlib')
    except ImportError as e:
        print("Tikzplotlib not installed, consider installing it to get more plotting capabilities")

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

    def __init__(self, root_dir: str = "/media/barrachina/data/results/Journal_MLSP/new"):
        """
        Finds all paths in a given `root_dir` directory
        :param root_dir:
        """
        child_dirs = os.walk(root_dir)
        monte_dict = defaultdict(lambda: defaultdict(list))
        for child_dir in child_dirs:
            if "run-" in child_dir[0].split('/')[-1]:
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
                                "balance": self._get_balance(simu_params)
                            }
                            if (Path(child_dir[0]) / 'evaluate.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["eval"].append(
                                    str(Path(child_dir[0]) / 'evaluate.csv'))
                            if (Path(child_dir[0]) / 'train_confusion_matrix.csv').is_file():
                                monte_dict[json.dumps(params, sort_keys=True)]["train_conf"].append(
                                    str(Path(child_dir[0]) / 'train_confusion_matrix.csv'))
                                monte_dict[json.dumps(params, sort_keys=True)]["val_conf"].append(
                                    str(Path(child_dir[0]) / 'val_confusion_matrix.csv'))
                            monte_dict[json.dumps(params, sort_keys=True)]["data"].append(
                                str(Path(child_dir[0]) / 'history_dict.csv'))
                            monte_dict[json.dumps(params, sort_keys=True)]["image"].append(
                                str(Path(child_dir[0]) / 'prediction.png'))
                            if not os.path.isfile(monte_dict[json.dumps(params, sort_keys=True)]["image"][-1]):
                                print("Generating picture predicted figure. "
                                      "This will take a while but will only be done once in a lifetime")
                                # If I dont have the image I generate it
                                dataset_name = params["dataset"].upper()
                                if dataset_name == "BRETIGNY":  # For version compatibility
                                    dataset_name = "BRET"
                                mode = "t" if 'coherency' in simu_params else "s"
                                dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode,
                                                                       complex_mode='real_mode' not in simu_params,
                                                                       normalize=False, real_mode="real_imag",
                                                                       balance=(params['balance'] == "dataset"))
                                get_final_model_results(Path(child_dir[0]), dataset_handler=dataset_handler,
                                                        model_name=params["model"],
                                                        tensorflow='tensorflow' in simu_params,
                                                        complex_mode='real_mode' not in simu_params,
                                                        channels=6 if 'coherency' in simu_params else 3,
                                                        weights=dataset_handler.weights if
                                                        params['balance'] == "loss" else None, dropout=None)
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
            pandas_dict = pd.DataFrame()
            for data_results_dict in self.monte_dict[json_key]['eval']:
                result_pandas = pd.read_csv(data_results_dict, index_col=0)
                pandas_dict = pd.concat([pandas_dict, result_pandas], sort=False)
            self.monte_dict[json_key]['eval_stats'] = pandas_dict.groupby(pandas_dict.index).describe()
        return self.monte_dict[json_key]['eval_stats']

    def get_conf_stats(self, json_key: str):
        if len(self.monte_dict[json_key]['conf_stats']) == 0:
            cm = []
            # TODO: Put separated function. Repeat code twice
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
            assert len(self.monte_dict[json_key]['conf_stats']) == 2
        return self.monte_dict[json_key]['conf_stats']

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
            next_value = simu_params.split()[real_mode_index + 1]
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

    def box_plot(self, labels: List[str], data: List, ax=None,
                 key='accuracy', library='seaborn', epoch=-1, showfig=False, savefile: Optional[str] = None):
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
            self._box_plot_plotly(labels=labels, mc_runs=data, key=key, epoch=epoch, showfig=showfig, savefile=savefile)
        # TODO: https://seaborn.pydata.org/examples/grouped_boxplot.html
        elif library == 'seaborn':
            self._box_plot_seaborn(labels=labels, data=data, key=key, epoch=epoch, showfig=showfig, savefile=savefile)
        else:
            raise ModuleNotFoundError(f"Library {library} requested for plotting unknown")
        return None

    def _box_plot_plotly(self, labels: List[str], mc_runs: List,
                          key='accuracy', epoch=-1, showfig=False, savefile=None, extension=".svg"):
        if 'plotly' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(f"No Plotly installed, function {self._box_plot_plotly.__name__} "
                                      f"was called but will be omitted")
        savefig = False
        if savefile is not None:
            savefig = True
        epochs = []
        for i in range(len(mc_runs)):
            if epoch == -1:
                epochs.append(max(mc_runs[i].epoch))  # get last epoch
            else:
                epochs.append(epoch)
        # Prepare data
        fig = go.Figure()
        # color_pal = []
        for i, mc_run in enumerate(mc_runs):
            # color_pal += sns.color_palette()[:len(df.network.unique())]
            filter = mc_run['epoch'] == epochs[i]
            data = mc_run[filter]
            fig.add_trace(go.Box(
                y=data[key],
                name=labels[i],
                whiskerwidth=0.2,
                notched=True,  # confidence intervals for the median
                fillcolor=add_transparency(DEFAULT_PLOTLY_COLORS[i], 0.5),
                boxpoints='suspectedoutliers',  # to mark the suspected outliers
                line=dict(color=DEFAULT_PLOTLY_COLORS[i]),
                boxmean=True  # Interesting how sometimes it falls outside the box
            ))
        fig.update_layout(
            title='Plotly Box Plot',
            yaxis=dict(
                title=key,
                autorange=True,
                showgrid=True,
                dtick=0.05,
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
                          key='accuracy', epoch=-1, showfig=False, savefile=None, extension=".svg"):
        if 'seaborn' not in AVAILABLE_LIBRARIES:
            raise ModuleNotFoundError(
                "No Seaborn installed, function " + self._box_plot_seaborn.__name__ + " was called but will be omitted")
        epochs = []
        for i in range(len(data)):
            if epoch == -1:
                epochs.append(max(data[i].epoch))    # get last epoch
            else:
                epochs.append(epoch)
        # Prepare data
        frames = []
        # color_pal = []
        for i, mc_run in enumerate(data):
            # color_pal += sns.color_palette()[:len(df.network.unique())]
            filter = mc_run['epoch'] == epochs[i]
            t_data = mc_run[filter]
            frames.append(t_data)
        result = pd.concat(frames)

        # Run figure
        fig = plt.figure()
        ax = sns.boxplot(x=labels, y=key, data=result, boxprops=dict(alpha=.3))
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

        if savefile is not None:
            if not savefile.endswith(extension):
                savefile += extension
            os.makedirs(os.path.split(savefile)[0], exist_ok=True)
            fig.savefig(savefile, transparent=True)
            if 'tikzplotlib' not in AVAILABLE_LIBRARIES:
                raise ModuleNotFoundError(
                    "No Tikzplotlib installed, function " + self._box_plot_seaborn.__name__ + " will not save tex file")
            else:
                tikzplotlib.save(Path(os.path.split(savefile)[0]) / ("tikz_box_plot.tex"))
        if showfig:
            fig.show()
        return fig, ax


if __name__ == "__main__":
    simulation_results = ResultReader(root_dir=
                                      "/media/barrachina/data/results/Journal_MLSP/old/During-Marriage-simulations")
    lst = list(simulation_results.monte_dict.keys())
    data = [simulation_results.get_pandas_data(lst[10])]
    data.append(simulation_results.get_pandas_data(lst[11]))
    SeveralMonteCarloPlotter().box_plot(labels=["one", "two"], data=data, showfig=True, library='plotly')
