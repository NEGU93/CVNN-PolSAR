import os
import sys
import random
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QSlider, QTableView, QHeaderView, QSizePolicy
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from typing import Dict, List
from cvnn.utils import REAL_CAST_MODES
from principal_simulation import save_result_image_from_saved_model, _get_dataset_handler

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


def _get_model(simu_params):
    try:
        model_index = simu_params.split().index('--model')
    except ValueError:
        model_index = -1
    return f"{simu_params.split()[model_index + 1] if model_index != -1 else 'cao'}".lower()


def _get_dataset(simu_params):
    try:
        dataset_index = simu_params.split().index('--dataset')
    except ValueError:
        dataset_index = -1
    return_value = f"{simu_params.split()[dataset_index + 1] if dataset_index != -1 else 'SF-AIRSAR'}".upper()
    if return_value == "BRETIGNY":
        return_value = "BRET"
    return return_value


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


def _get_dataset_method(simu_params):
    try:
        dataset_method_index = simu_params.split().index('--dataset_method')
    except ValueError:
        dataset_method_index = -1
    return f"{simu_params.split()[dataset_method_index + 1] if dataset_method_index != -1 else 'random'}"


def _get_real_mode(simu_params):
    if 'real_mode' in simu_params:
        real_mode_index = simu_params.split().index('--real_mode')
        next_value = simu_params.split()[real_mode_index + 1]
        return next_value if next_value in REAL_CAST_MODES else 'real_imag'
    elif 'tensorflow' in simu_params:
        return 'real_imag'
    else:
        return 'complex'


def get_paths(root_dir: str = "/media/barrachina/data/results/During-Marriage-simulations") -> dict:
    """
    Finds all paths in a given `root_dir` directory
    :param root_dir:
    :return:
    """
    child_dirs = os.walk(root_dir)
    monte_dict = {}
    for child_dir in child_dirs:
        if "run-" in child_dir[0].split('/')[-1]:
            file_path = Path(child_dir[0]) / "model_summary.txt"
            if file_path.is_file():
                with open(file_path) as txt_sum_file:
                    simu_params = txt_sum_file.readline()
                    if (Path(child_dir[0]) / 'history_dict.csv').is_file():
                        # TODO: Verify model and other strings are valid
                        monte_dict[child_dir[0]] = {
                            "data": str(Path(child_dir[0]) / 'history_dict.csv'),
                            "image": str(Path(child_dir[0]) / 'prediction.png'),
                            "params": {
                                "dataset": _get_dataset(simu_params), "model": _get_model(simu_params),
                                "dtype": _get_real_mode(simu_params),
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset_method": _get_dataset_method(simu_params),
                                "balance": _get_balance(simu_params)
                            }
                        }
                        if not os.path.isfile(monte_dict[child_dir[0]]["image"]):
                            # If I dont have the image I generate it
                            dataset_name = monte_dict[child_dir[0]]["params"]["dataset"].upper()
                            if dataset_name == "BRETIGNY":
                                dataset_name = "BRET"
                            mode = "t" if 'coherency' in simu_params else "s"
                            dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode,
                                                                   complex_mode='real_mode' not in simu_params,
                                                                   normalize=False, real_mode="real_imag",
                                                                   balance=(monte_dict[child_dir[0]]["params"][
                                                                                'balance'] == "dataset"))
                            save_result_image_from_saved_model(Path(child_dir[0]), dataset_handler=dataset_handler,
                                                               model_name=monte_dict[child_dir[0]]["params"]["model"],
                                                               tensorflow='tensorflow' in simu_params,
                                                               complex_mode='real_mode' not in simu_params,
                                                               channels=6 if 'coherency' in simu_params else 3,
                                                               weights=monte_dict[child_dir[0]]["params"] if
                                                               monte_dict[child_dir[0]]["params"] != "None" else None)
                    else:
                        print("No history_dict found on path " + child_dir[0])
            else:
                print("No model_summary.txt found in " + child_dir[0])
    return monte_dict


"""
    MonteCarlo Plotter
"""


class MonteCarloPlotter:

    def plot(self, paths: List[str], keys: List[str], ax=None):
        """
        :param paths: list of history dictionaries (output of Model.fit())
        :param ax: (Optional) axis on which to plot the data
        :param keys:
        :return:
        """
        pandas_dict = pd.DataFrame()
        for data_results_dict in paths:
            result_pandas = pd.read_csv(data_results_dict, index_col=False)
            pandas_dict = pd.concat([pandas_dict, result_pandas], sort=False)
        self._plot_line_confidence_interval_matplotlib(ax=ax, keys=keys, data=pandas_dict)
        return pandas_dict

    def _plot_line_confidence_interval_matplotlib(self, keys: List[str], data, ax=None, showfig=False, x_axis='epoch'):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        for i, key in enumerate(keys):
            x = data[x_axis].unique().tolist()
            # set_trace()
            stats = data.groupby('epoch').describe()
            data_mean = stats[key]['mean'].tolist()
            data_max = stats[key]['max'].tolist()
            data_min = stats[key]['min'].tolist()
            data_50 = stats[key]['50%'].tolist()
            data_25 = stats[key]['25%'].tolist()
            data_75 = stats[key]['75%'].tolist()
            ax.plot(x, data_mean, color=DEFAULT_MATPLOTLIB_COLORS[i%len(DEFAULT_MATPLOTLIB_COLORS)],
                    label=key)
            ax.plot(x, data_50, '--', color=DEFAULT_MATPLOTLIB_COLORS[i%len(DEFAULT_MATPLOTLIB_COLORS)])
                    # label=key + ' median')
            ax.fill_between(x, data_25, data_75, color=DEFAULT_MATPLOTLIB_COLORS[i%len(DEFAULT_MATPLOTLIB_COLORS)],
                            alpha=.4)   # , label=key + ' interquartile')
            ax.fill_between(x, data_min, data_max, color=DEFAULT_MATPLOTLIB_COLORS[i%len(DEFAULT_MATPLOTLIB_COLORS)],
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
        self.setWindowTitle("Results")
        self.label_image = QLabel()
        self.image_paths = get_paths()
        self.params = START_VALUES  # start config to show
        self.params_label = QLabel(str(self.params))
        self.params_label.setAlignment(Qt.AlignCenter)
        self.df = pd.DataFrame()
        self.plotter = MonteCarloPlotter()
        for img in self.image_paths.values():
            self.df = pd.concat([self.df, pd.DataFrame(img['params'], index=[0])], ignore_index=True)
        self.df.sort_values(by=['dataset', 'model', 'dtype', 'library', 'dataset_mode', 'dataset_method', 'balance'])
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
        hlayout.addLayout(self.radiobuttons())
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
        self.tableView.setModel(DataFrameModel(self.df))
        self.tableView.setAlternatingRowColors(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(len(self.df.keys())):
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
        self.coh_rb.toggled.connect(lambda: self.update_image("dataset_mode", self.coh_rb.text()))

        rb2 = QRadioButton("k", self)
        rb2.toggled.connect(lambda: self.update_image("dataset_mode", rb2.text()))

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
        rb1.toggled.connect(lambda state: state and self.update_image("dtype", rb1.text()))

        self.real_dtype_rb = QRadioButton("real_imag", self)
        self.real_dtype_rb.toggled.connect(lambda state: state and self.update_image("dtype", self.real_dtype_rb.text()))

        rb2 = QRadioButton("amplitude_phase")
        rb2.toggled.connect(lambda state: state and self.update_image("dtype", rb1.text()))
        rb3 = QRadioButton("amplitude_only")
        rb3.toggled.connect(lambda state: state and self.update_image("dtype", rb1.text()))
        rb4 = QRadioButton("real_only")
        rb4.toggled.connect(lambda state: state and self.update_image("dtype", rb1.text()))

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
        rb1.toggled.connect(lambda: self.update_image("balance", rb1.text()))

        rb2 = QRadioButton("loss", self)
        rb2.toggled.connect(lambda: self.update_image("balance", rb2.text()))

        rb3 = QRadioButton("dataset")
        rb3.toggled.connect(lambda: self.update_image("balance", rb3.text()))

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
        self.cvnn_library_rb.toggled.connect(lambda state: state and self.update_image("library", self.cvnn_library_rb.text()))

        rb2 = QRadioButton("tensorflow", self)
        rb2.toggled.connect(lambda state: state and self.update_image("library", rb2.text()))

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
        rb1.toggled.connect(lambda: self.update_image("dataset_method", rb1.text()))

        rb2 = QRadioButton("separate", self)
        rb2.toggled.connect(lambda: self.update_image("dataset_method", rb2.text()))

        rb3 = QRadioButton("single_separated_image")
        rb3.toggled.connect(lambda: self.update_image("dataset_method", rb3.text()))


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
        rb1.toggled.connect(lambda: self.update_image("model", rb1.text()))

        rb2 = QRadioButton("own", self)
        rb2.toggled.connect(lambda: self.update_image("model", rb2.text()))

        rb3 = QRadioButton("zhang")
        rb3.toggled.connect(lambda: self.update_image("model", rb3.text()))

        rb4 = QRadioButton("haensch", self)
        rb4.toggled.connect(lambda: self.update_image("model", rb4.text()))

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
        rb1.toggled.connect(lambda: self.update_image("dataset", rb1.text()))

        rb2 = QRadioButton("SF-RS2", self)
        rb2.toggled.connect(lambda: self.update_image("dataset", rb2.text()))

        rb3 = QRadioButton("OBER")
        rb3.toggled.connect(lambda: self.update_image("dataset", rb3.text()))

        rb4 = QRadioButton("BRET", self)
        rb4.toggled.connect(lambda: self.update_image("dataset", rb4.text()))

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
        if len(history_path) == 1:
            history_path = history_path[0]
            if os.path.isfile(history_path):
                data_pd = pd.read_csv(history_path)
                # import pdb; pdb.set_trace()
                data_pd.plot(ax=ax1, x="epoch", y=["accuracy", "val_accuracy"])
                data_pd.plot(ax=ax2, x="epoch", y=["loss", "val_loss"])
                ax1.set_xlim(left=0, right=max(data_pd['epoch']))
                ax2.set_xlim(left=0, right=max(data_pd['epoch']))

                # import pdb; pdb.set_trace()
                self.acc_values[0].setText(f"{data_pd['accuracy'].iloc[-1]:.2%}")
                self.acc_values[1].setText(f"{data_pd['average_accuracy'].iloc[-1]:.2%}")
                self.acc_values[2].setText(f"{data_pd['val_accuracy'].iloc[-1]:.2%}")
                self.acc_values[3].setText(f"{data_pd['val_average_accuracy'].iloc[-1]:.2%}")
            elif hasattr(self, 'acc_values'):
                self.acc_values[0].setText("00.00%")
                self.acc_values[1].setText("00.00%")
                self.acc_values[2].setText("00.00%")
                self.acc_values[3].setText("00.00%")
        elif history_path:  # Not empty
            data_pd = self.plotter.plot(paths=history_path, ax=ax1, keys=["accuracy", "val_accuracy"])
            self.plotter.plot(paths=history_path, ax=ax2, keys=["loss", "val_loss"])
            self.acc_values[0].setText(f"{data_pd.groupby('epoch').describe()['accuracy']['mean'].iloc[-1]:.2%}")
            self.acc_values[1].setText(f"{data_pd.groupby('epoch').describe()['average_accuracy']['mean'].iloc[-1]:.2%}")
            self.acc_values[2].setText(f"{data_pd.groupby('epoch').describe()['val_accuracy']['mean'].iloc[-1]:.2%}")
            self.acc_values[3].setText(f"{data_pd.groupby('epoch').describe()['val_average_accuracy']['mean'].iloc[-1]:.2%}")
        ax1.grid(True, axis='both')
        ax2.grid(True, axis='both')
        self.canvas.draw()

    def update_image(self, key, value):
        self.params[key] = value
        self.params_label.setText(str(self.params))
        # Not yet working. Try https://stackoverflow.com/questions/49929668/disable-and-enable-radiobuttons-from-another-radiobutton-in-pyqt4-python
        if value == "complex":
            if hasattr(self, 'cvnn_library_rb'):    # It wont exists if I still didnt create the radiobutton.
                self.cvnn_library_rb.setChecked(True)   # Set library cvnn
        elif value == "tensorflow":
            if hasattr(self, 'real_dtype_rb'):
                self.real_dtype_rb.setChecked(True)   # Set real dtype
        elif value == "OBER":
            if hasattr(self, 'coh_rb'):
                self.coh_rb.setChecked(True)   # Set real dtype
        path = []
        hist = []
        for img_path in self.image_paths.values():
            if self._compare_params(img_path['params']):
                path.append(img_path['image'])
                hist.append(img_path['data'])
        self.get_image(path)
        self.plot(hist)

    def _compare_params(self, param: Dict[str, str]) -> bool:
        for key in param:
            if param[key] != self.params[key]:
                return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
