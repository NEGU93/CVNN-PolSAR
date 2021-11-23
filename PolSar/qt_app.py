import os
import sys
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QSlider, QTableView
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from typing import Dict, List
from cvnn.utils import REAL_CAST_MODES
from principal_simulation import save_result_image_from_saved_model, _get_dataset_handler

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
        balance_index = simu_params.split().index('--model')
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
        self.df = pd.DataFrame()
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

    def _get_upper_layout(self):
        hlayout = QHBoxLayout()  # Main layout. Horizontal 2 things, radio buttons + image
        hlayout.addLayout(self.radiobuttons())
        img_layout = QVBoxLayout()
        img_layout.addWidget(self.label_image)  # Show image
        hlayout.addLayout(img_layout)
        return hlayout

    def _get_lower_layout(self):
        hlayout = QHBoxLayout()

        hlayout.addLayout(self._get_dataframe_table_layout())
        hlayout.addLayout(self._get_figure_layout())

        return hlayout

    def _get_dataframe_table_layout(self):
        vlayout = QVBoxLayout()
        self.params_label = QLabel(str(self.params))
        self.params_label.setAlignment(Qt.AlignCenter)
        self.tableView = QTableView()
        # self.verticalLayout.addWidget(self.tableView)
        self.tableView.setModel(DataFrameModel(self.df))
        vlayout.addWidget(self.params_label)  # Current config
        vlayout.addWidget(self.tableView)
        return vlayout

    def _get_figure_layout(self):
        self.figure = plt.figure()
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
        return vlayout

    def dataset_mode_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("coh")
        rb1.toggled.connect(lambda: self.update_image("dataset_mode", rb1.text()))

        rb2 = QRadioButton("k", self)
        rb2.toggled.connect(lambda: self.update_image("dataset_mode", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addStretch()

        return vlayout

    def dtype_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("complex")
        rb1.toggled.connect(lambda: self.update_image("dtype", rb1.text()))

        self.real_dtype_rb = QRadioButton("real_imag", self)
        self.real_dtype_rb.toggled.connect(lambda: self.update_image("dtype", self.real_dtype_rb.text()))

        rb2 = QRadioButton("amplitude_phase")
        rb2.toggled.connect(lambda: self.update_image("dtype", rb1.text()))
        rb3 = QRadioButton("amplitude_only")
        rb3.toggled.connect(lambda: self.update_image("dtype", rb1.text()))
        rb4 = QRadioButton("real_only")
        rb4.toggled.connect(lambda: self.update_image("dtype", rb1.text()))

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
        self.cvnn_library_rb.toggled.connect(lambda: self.update_image("library", self.cvnn_library_rb.text()))

        rb2 = QRadioButton("tensorflow", self)
        rb2.toggled.connect(lambda: self.update_image("library", rb2.text()))

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

    def get_image(self, image_path: str):
        if image_path == '':
            pixmap = QPixmap(BASE_PATHS[self.params["dataset"]])
        else:
            pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1000, 700, QtCore.Qt.KeepAspectRatio)
        self.label_image.setPixmap(scaled_pixmap)
        self.label_image.resize(scaled_pixmap.width(), scaled_pixmap.height())

        # self.show()

    def plot(self, history_path):
        self.figure.clear()
        if os.path.isfile(history_path):
            data_pd = pd.read_csv(history_path)
            # import pdb; pdb.set_trace()
            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)
            ax1.clear()
            ax2.clear()
            data_pd.plot(ax=ax1, x="epoch", y=["accuracy", "val_accuracy"])
            data_pd.plot(ax=ax2, x="epoch", y=["loss", "val_loss"])
            ax1.grid(True, axis='both')
            ax2.grid(True, axis='both')
            ax1.set_xlim(left=0, right=max(data_pd['epoch']))
            ax2.set_xlim(left=0, right=max(data_pd['epoch']))
            self.canvas.draw()

    def update_image(self, key, value):
        self.params[key] = value
        self.params_label.setText(str(self.params))
        # Not yet working. Try https://stackoverflow.com/questions/49929668/disable-and-enable-radiobuttons-from-another-radiobutton-in-pyqt4-python
        # if value == "complex":
        #     if hasattr(self, 'cvnn_library_rb'):    # It wont exists if I still didnt create the radiobutton.
        #         self.cvnn_library_rb.setChecked(True)   # Set library cvnn
        #         self.params["library"] = "cvnn"
        # elif value == "tensorflow":
        #     if hasattr(self, 'real_dtype_rb'):
        #         self.real_dtype_rb.setChecked(True)   # Set real dtype
        #         self.params["dtype"] = "real"
        path = ""
        hist = ""
        for img_path in self.image_paths.values():
            if self._compare_params(img_path['params']):
                path = img_path['image']
                hist = img_path['data']
                break
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
