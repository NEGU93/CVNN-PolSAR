import os
import sys
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QSlider, QTableView
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from typing import Dict, List
from principal_simulation import DATASET_META, MODEL_META, save_result_image_from_saved_model, _get_dataset_handler

BASE_PATHS = {
    "BRET": "/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/bret-2003.png",
    "OBER": "/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6/PauliRGB_T1.bmp",
    "SF-AIRSAR": "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF/SF-AIRSAR/SF-AIRSAR-Pauli.bmp",
    "SF-RS2": "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF/SF-RS2/SF-RS2-Pauli.bmp"
}
START_VALUES = {
    "dataset_method": 'random',
    "library": "complex",
    "model": "cao",
    "balance": 'None',
    "dtype": 'complex',
    "dataset_mode": 'k',
    "dataset": 'SF-AIRSAR'
}


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
                        try:
                            dataset_method_index = simu_params.split().index('--dataset_method')
                        except ValueError:
                            dataset_method_index = -1
                        try:
                            model_index = simu_params.split().index('--model')
                        except ValueError:
                            model_index = -1
                        try:
                            balance_index = simu_params.split().index('--balance')
                        except ValueError:
                            balance_index = -1
                        try:
                            dataset_index = simu_params.split().index('--dataset')
                        except ValueError:
                            dataset_index = -1
                        # TODO: Verify model and other strings are valid
                        monte_dict[child_dir[0]] = {
                            "data": str(Path(child_dir[0]) / 'history_dict'),
                            "image": str(Path(child_dir[0]) / 'prediction.png'),
                            "params": {
                                "dataset_method": f"{simu_params.split()[dataset_method_index + 1] if dataset_method_index != -1 else 'random'}",
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "model": f"{simu_params.split()[model_index + 1] if model_index != -1 else 'cao'}",
                                "balance": f"{simu_params.split()[balance_index + 1] if balance_index != -1 else 'None'}",
                                "dtype": f"{'real' if 'real_mode' in simu_params else 'complex'}",
                                # TODO: No detail of real mode
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset": f"{simu_params.split()[dataset_index + 1] if dataset_index != -1 else 'SF-AIRSAR'}"
                            }
                        }
                        if not os.path.isfile(monte_dict[child_dir[0]]["image"]):     # If I dont have the image I generate it
                            dataset_name = monte_dict[child_dir[0]]["params"]["dataset"].upper()
                            if dataset_name == "BRETIGNY":
                                dataset_name = "BRET"
                            mode = "t" if 'coherency' in simu_params else "k"
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
        self.params_label = QLabel(str(self.params))
        df = pd.DataFrame()
        for img in self.image_paths.values():
            df = pd.concat([df, pd.DataFrame(img['params'], index=[0])], ignore_index=True)
        # df.sort_values(by=['labels', 'dtype', 'dataset_mode', 'dataset_split', 'loss', 'boxcar'])
        self.btngroup = []
        widget = QWidget()
        hlayout = QHBoxLayout()  # Main layout. Horizontal 2 things, radio buttons + image

        self.get_image('')
        hlayout.addLayout(self.radiobuttons())

        img_layout = QVBoxLayout()
        self.params_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.params_label)  # Current config
        img_layout.addWidget(self.label_image)  # Show image
        hlayout.addLayout(img_layout)

        outer_layout = QVBoxLayout()
        outer_layout.addLayout(hlayout)
        self.tableView = QTableView()
        # self.verticalLayout.addWidget(self.tableView)
        self.tableView.setModel(DataFrameModel(df))
        outer_layout.addWidget(self.tableView)
        widget.setLayout(outer_layout)
        # widget.setLayout(hlayout)
        self.setCentralWidget(widget)
        self.show()

    def radiobuttons(self):
        vlayout = QVBoxLayout()

        vlayout.addLayout(self.add_title(self.add_radiobuttons('dataset', list(DATASET_META.keys())), "Dataset"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('model', list(MODEL_META.keys())), "Model"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('dataset_method', ["random", "separate",
                                                                                  "single_separated_image"]),
                                         "Dataset Method"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('library', ["tensorflow", "cvnn"]), "Library"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('balance', ["None", "loss", "dataset"]), "Balance"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('dtype', ["complex", "real"]), "Dtype"))
        vlayout.addLayout(self.add_title(self.add_radiobuttons('dataset_mode', ["coh", "k"]), "Dataset Mode"))
        return vlayout

    def add_title(self, layout, name: str, up: bool = True):
        lay = QVBoxLayout()
        l1 = QLabel(name)
        l1.setAlignment(Qt.AlignCenter)
        if up:
            lay.addWidget(l1)
        lay.addLayout(layout)
        if not up:
            lay.addWidget(l1)
        return lay

    def add_radiobuttons(self, name: str, options: List[str]):
        vlayout = QHBoxLayout()
        radio_buttons = []
        for option in options:
            rb = QRadioButton(option, self)
            rb.toggled.connect(lambda: self.update_image(name, rb.text()))
            radio_buttons.append(rb)
        self.btngroup.append(QButtonGroup())
        for button in radio_buttons:
            self.btngroup[-1].addButton(button)
        radio_buttons[0].setChecked(True)
        for button in radio_buttons:
            vlayout.addWidget(button)
        return vlayout

    def get_image(self, image_path: str):
        if image_path == '':
            pixmap = QPixmap(BASE_PATHS[self.params["dataset"]])
        else:
            pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaledToWidth(1000)
        self.label_image.setPixmap(scaled_pixmap)
        self.label_image.resize(scaled_pixmap.width(), scaled_pixmap.height())
        # self.show()

    def update_image(self, key, value):
        self.params[key] = value
        self.params_label.setText(str(self.params))
        # if value == "tensorflow":
        #     self.rb.setChecked(True)
        #     self.params["dtype"] = "real"       # Not sure if this is necessary but it wont hurt
        path = ""
        for img_path in self.image_paths.values():
            if self._compare_params(img_path['params']):
                path = img_path['image']
                break
        self.get_image(path)

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
