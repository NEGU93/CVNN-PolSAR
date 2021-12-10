import os
import sys
import random
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from seaborn import heatmap
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QTableView, QHeaderView, QSizePolicy
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from typing import List
from cvnn.utils import REAL_CAST_MODES
from principal_simulation import get_final_model_results, _get_dataset_handler

from pdb import set_trace

BASE_PATHS = {
    "BRET": "/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/bret-2003.png",
    "OBER": "/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6/PauliRGB_T1.bmp",
    "SF-AIRSAR": "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF/SF-AIRSAR/SF-AIRSAR-Pauli.bmp",
    "SF-RS2": "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF/SF-RS2/SF-RS2-Pauli.bmp"
}
START_VALUES = {
    "dataset": 'SF-AIRSAR',
    "model": "cao",
    "dtype": 'complex',
    "library": "complex",
    "dataset_mode": 'k',
    "dataset_method": 'random',
    "balance": 'None',
}

DEFAULT_MATPLOTLIB_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


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

    def data_exists(self, json_key: str):
        return bool(self.monte_dict[json_key]['data'])

    def get_stats(self, json_key: str):
        """
        The 'stats' key is lazy. It will only be calculated on the first call to this function
        """
        if len(self.monte_dict[json_key]['stats']) == 0:
            pandas_dict = pd.DataFrame()
            for data_results_dict in self.monte_dict[json_key]['data']:
                result_pandas = pd.read_csv(data_results_dict, index_col=False)
                pandas_dict = pd.concat([pandas_dict, result_pandas], sort=False)
            self.monte_dict[json_key]['stats'] = pandas_dict.groupby('epoch').describe()
        return self.monte_dict[json_key]['stats']

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


"""
    Qt methods
"""


class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
                                       and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

    def flags(self, index) -> Qt.ItemFlags:
        # QtCore.Qt.ItemIsEditable |
        # | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
        # | QtCore.Qt.ItemIsTristate
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variables
        self.simulation_results = ResultReader()
        self.params = START_VALUES  # start config to show
        self.plotter = MonteCarloPlotter()

        # Qt objects
        self.setWindowTitle("Results")
        self.label_image = QLabel()
        self.params_label = QLabel(str(self.params))
        self.params_label.setAlignment(Qt.AlignCenter)
        self.btngroup = []

        widget = QWidget()

        outer_layout = QVBoxLayout()
        lower_layout = self._get_lower_layout()
        outer_layout.addLayout(self._get_upper_layout())
        outer_layout.addLayout(lower_layout)
        widget.setLayout(outer_layout)
        # widget.setLayout(hlayout)
        self.setCentralWidget(widget)
        self.show()

    def _get_accuracy_layout(self):
        title = QLabel("Accuracy")
        myFont = QFont()
        myFont.setBold(True)
        title.setFont(myFont)
        key = []
        key.append(QLabel("Train OA: "))
        key.append(QLabel("Train AA: "))
        key.append(QLabel("Validation OA: "))
        key.append(QLabel("Validation AA: "))
        for k in key:
            k.setFont(myFont)
        self.acc_values = []
        self.acc_values.append(QLabel("00.00%"))
        self.acc_values.append(QLabel("00.00%"))
        self.acc_values.append(QLabel("00.00%"))
        self.acc_values.append(QLabel("00.00%"))
        vbox = QVBoxLayout()
        vbox.addWidget(title)
        for i in range(len(key)):
            hbox = QHBoxLayout()
            hbox.addWidget(key[i])
            hbox.addWidget(self.acc_values[i])
            hbox.setAlignment(Qt.AlignLeft)
            vbox.addLayout(hbox)
        return vbox

    def _get_upper_layout(self):
        hlayout = QHBoxLayout()  # Main layout. Horizontal 2 things, radio buttons + image
        vlayout = QVBoxLayout()
        vlayout.addLayout(self.radiobuttons())
        vlayout.addWidget(self._get_conf_figure_widget())
        hlayout.addLayout(vlayout)
        img_layout = QVBoxLayout()
        img_layout.addWidget(self.params_label)  # Current config
        img_layout.addWidget(self.label_image)  # Show image
        hlayout.addLayout(img_layout)
        return hlayout

    def _get_lower_layout(self):
        hlayout = QHBoxLayout()

        hlayout.addWidget(self._get_dataframe_table_layout())
        hlayout.addLayout(self._get_figure_layout())

        return hlayout

    def _get_dataframe_table_layout(self):
        # layout = QHBoxLayout()
        self.tableView = QTableView()
        # self.verticalLayout.addWidget(self.tableView)
        # set_trace()
        self.tableView.setModel(
            DataFrameModel(self.simulation_results.df
                           .drop_duplicates()
                           .sort_values(by=['dataset', 'model', 'dtype', 'library',
                                            'dataset_mode', 'dataset_method', 'balance'])
                           .reset_index(drop=True)
                           )
        )
        self.tableView.setAlternatingRowColors(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(len(self.simulation_results.df.keys())):
            self.tableView.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.tableView.setMinimumWidth(600)
        sp = self.tableView.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Minimum)
        self.tableView.setSizePolicy(sp)
        # layout.addWidget(self.tableView)
        return self.tableView

    def _get_figure_layout(self):
        self.figure = plt.figure()
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vlay = QVBoxLayout()
        vlay.addWidget(self.toolbar)
        vlay.addWidget(self.canvas)
        return vlay

    def _get_conf_figure_widget(self):
        self.conf_figure = plt.figure()
        self.conf_figure.tight_layout()
        self.conf_canvas = FigureCanvas(self.conf_figure)
        return self.conf_canvas

    def radiobuttons(self):
        vlayout = QVBoxLayout()

        vlayout.addLayout(self.add_title(self.dataset_radiobutton(), name="Dataset"))
        vlayout.addLayout(self.add_title(self.model_radiobutton(), "Model"))
        vlayout.addLayout(self.add_title(self.dtype_radiobuttons(), "Dtype"))
        vlayout.addLayout(self.add_title(self.library_radiobutton(), "Library"))
        vlayout.addLayout(self.add_title(self.dataset_mode_radiobuttons(), "Dataset Mode"))
        vlayout.addLayout(self.add_title(self.model_method_radiobutton(), "Dataset Method"))
        vlayout.addLayout(self.add_title(self.balance_radiobuttons(), "Balance"))
        vlayout.addStretch()
        vlayout.addLayout(self._get_accuracy_layout())
        return vlayout

    def dataset_mode_radiobuttons(self):
        vlayout = QHBoxLayout()
        self.coh_rb = QRadioButton("coh")
        self.coh_rb.toggled.connect(lambda: self.update_information("dataset_mode", self.coh_rb.text()))

        rb2 = QRadioButton("k", self)
        rb2.toggled.connect(lambda: self.update_information("dataset_mode", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(self.coh_rb)

        self.coh_rb.setChecked(True)
        vlayout.addWidget(self.coh_rb)
        vlayout.addWidget(rb2)
        vlayout.addStretch()

        return vlayout

    def dtype_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("complex")
        rb1.toggled.connect(lambda state: state and self.update_information("dtype", rb1.text()))

        self.real_dtype_rb = QRadioButton("real_imag", self)
        self.real_dtype_rb.toggled.connect(
            lambda state: state and self.update_information("dtype", self.real_dtype_rb.text()))

        rb2 = QRadioButton("amplitude_phase")
        rb2.toggled.connect(lambda state: state and self.update_information("dtype", rb1.text()))
        rb3 = QRadioButton("amplitude_only")
        rb3.toggled.connect(lambda state: state and self.update_information("dtype", rb1.text()))
        rb4 = QRadioButton("real_only")
        rb4.toggled.connect(lambda state: state and self.update_information("dtype", rb1.text()))

        self.btngroup.append(QButtonGroup())
        # self.btngroup[-1].addButton(self.real_dtype_rb)
        self.btngroup[-1].addButton(rb1)
        self.btngroup[-1].addButton(self.real_dtype_rb)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb4)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(self.real_dtype_rb)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addWidget(rb4)
        vlayout.addStretch()

        return vlayout

    def balance_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("none")
        rb1.toggled.connect(lambda: self.update_information("balance", rb1.text()))

        rb2 = QRadioButton("loss", self)
        rb2.toggled.connect(lambda: self.update_information("balance", rb2.text()))

        rb3 = QRadioButton("dataset")
        rb3.toggled.connect(lambda: self.update_information("balance", rb3.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addStretch()

        return vlayout

    def library_radiobutton(self):
        vlayout = QHBoxLayout()
        self.cvnn_library_rb = QRadioButton("cvnn", self)
        self.cvnn_library_rb.toggled.connect(
            lambda state: state and self.update_information("library", self.cvnn_library_rb.text()))

        rb2 = QRadioButton("tensorflow", self)
        rb2.toggled.connect(lambda state: state and self.update_information("library", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(self.cvnn_library_rb)

        self.cvnn_library_rb.setChecked(True)
        vlayout.addWidget(self.cvnn_library_rb)
        vlayout.addWidget(rb2)
        vlayout.addStretch()

        return vlayout

    def model_method_radiobutton(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("random")
        rb1.toggled.connect(lambda: self.update_information("dataset_method", rb1.text()))

        rb2 = QRadioButton("separate", self)
        rb2.toggled.connect(lambda: self.update_information("dataset_method", rb2.text()))

        rb3 = QRadioButton("single_separated_image")
        rb3.toggled.connect(lambda: self.update_information("dataset_method", rb3.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addStretch()

        return vlayout

    def model_radiobutton(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("cao")
        rb1.toggled.connect(lambda: self.update_information("model", rb1.text()))

        rb2 = QRadioButton("own", self)
        rb2.toggled.connect(lambda: self.update_information("model", rb2.text()))

        rb3 = QRadioButton("zhang")
        rb3.toggled.connect(lambda: self.update_information("model", rb3.text()))

        rb4 = QRadioButton("haensch", self)
        rb4.toggled.connect(lambda: self.update_information("model", rb4.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb4)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addWidget(rb4)
        vlayout.addStretch()

        return vlayout

    def dataset_radiobutton(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("SF-AIRSAR")
        rb1.toggled.connect(lambda: self.update_information("dataset", rb1.text()))

        rb2 = QRadioButton("SF-RS2", self)
        rb2.toggled.connect(lambda: self.update_information("dataset", rb2.text()))

        rb3 = QRadioButton("OBER")
        rb3.toggled.connect(lambda: self.update_information("dataset", rb3.text()))

        rb4 = QRadioButton("BRET", self)
        rb4.toggled.connect(lambda: self.update_information("dataset", rb4.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb4)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addWidget(rb4)
        vlayout.addStretch()

        return vlayout

    def add_title(self, layout, name: str, up: bool = True):
        lay = QVBoxLayout()
        l1 = QLabel(name)
        myFont = QFont()
        myFont.setBold(True)
        l1.setFont(myFont)
        l1.setAlignment(Qt.AlignLeft)
        if up:
            lay.addWidget(l1)
        lay.addLayout(layout)
        # lay.addStretch()
        if not up:
            lay.addWidget(l1)
        return lay

    def get_image(self, image_path: List[str]):
        if not image_path:
            pixmap = QPixmap(BASE_PATHS[self.params["dataset"]])
        else:
            image_path = random.choice(image_path)
            pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1000, 700, QtCore.Qt.KeepAspectRatio)
        self.label_image.setPixmap(scaled_pixmap)
        self.label_image.resize(scaled_pixmap.width(), scaled_pixmap.height())

    def plot(self, history_path):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        ax1.clear()
        ax2.clear()
        self.plotter.plot(data=history_path, ax=ax1, keys=["accuracy", "val_accuracy"])
        self.plotter.plot(data=history_path, ax=ax2, keys=["loss", "val_loss"])
        ax1.grid(True, axis='both')
        ax2.grid(True, axis='both')
        self.canvas.draw()

    def plot_conf_matrix(self, conf_list):
        self.conf_figure.clear()
        ax1 = self.conf_figure.add_subplot(121)
        ax2 = self.conf_figure.add_subplot(122)
        ax1.clear()
        ax2.clear()
        heatmap(conf_list[0].drop('Total', axis=1).drop('Total'), ax=ax1, annot=True, cmap="rocket_r")
        heatmap(conf_list[1].drop('Total', axis=1).drop('Total'), ax=ax2, annot=True, cmap="rocket_r")
        self.conf_canvas.draw()

    def print_values(self, history_path):
        if history_path is not None and len(history_path) != 0 and hasattr(self, "acc_values"):
            self.acc_values[0].setText(f"{history_path['train']['mean']['accuracy']:.2%} +- "
                                       f"{history_path['train']['std']['accuracy'] / history_path['train']['count']['accuracy']:.2%}")
            self.acc_values[1].setText(f"{history_path['train']['mean']['average_accuracy']:.2%} +- "
                                       f"{history_path['train']['std']['average_accuracy'] / history_path['train']['count']['average_accuracy']:.2%}")
            self.acc_values[2].setText(f"{history_path['val']['mean']['accuracy']:.2%} +- "
                                       f"{history_path['val']['std']['accuracy'] / history_path['val']['count']['accuracy']:.2%}")
            self.acc_values[3].setText(f"{history_path['val']['mean']['average_accuracy']:.2%} +- "
                                       f"{history_path['val']['std']['average_accuracy'] / history_path['val']['count']['average_accuracy']:.2%}")
        elif hasattr(self, "acc_values"):
            self.acc_values[0].setText(f"00.00%")
            self.acc_values[1].setText(f"00.00%")
            self.acc_values[2].setText(f"00.00%")
            self.acc_values[3].setText(f"00.00%")

    def update_information(self, key, value):
        self.params[key] = value
        self.params_label.setText(str(self.params))
        # Not yet working. Try https://stackoverflow.com/questions/49929668/disable-and-enable-radiobuttons-from-another-radiobutton-in-pyqt4-python
        if value == "complex":
            if hasattr(self, 'cvnn_library_rb'):  # It wont exists if I still didnt create the radiobutton.
                self.cvnn_library_rb.setChecked(True)  # Set library cvnn
        elif value == "tensorflow":
            if hasattr(self, 'real_dtype_rb'):
                self.real_dtype_rb.setChecked(True)  # Set real dtype
        elif value == "OBER":
            if hasattr(self, 'coh_rb'):
                self.coh_rb.setChecked(True)  # Set real dtype
        json_key = json.dumps(self.params, sort_keys=True)
        self.get_image(self.simulation_results.get_image(json_key))
        if self.simulation_results.data_exists(json_key):
            stats = self.simulation_results.get_stats(json_key=json_key)
            eval_stats = self.simulation_results.get_eval_stats(json_key=json_key)
            conf_stats = self.simulation_results.get_conf_stats(json_key=json_key)
            self.plot(stats)
            self.print_values(eval_stats)
            self.plot_conf_matrix(conf_stats)
        else:
            self.print_values(None)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
