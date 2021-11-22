import os
import sys
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QSlider, QTableView
from PyQt5.QtGui import QPixmap, QFont
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
                                "dataset": f"{simu_params.split()[dataset_index + 1] if dataset_index != -1 else 'SF-AIRSAR'}",
                                "model": f"{simu_params.split()[model_index + 1] if model_index != -1 else 'cao'}",
                                "dtype": f"{'real' if 'real_mode' in simu_params else 'complex'}",
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset_method": f"{simu_params.split()[dataset_method_index + 1] if dataset_method_index != -1 else 'random'}",
                                "balance": f"{simu_params.split()[balance_index + 1].lower() if balance_index != -1 else 'none'}",
                            }
                        }
                        if monte_dict[child_dir[0]]["params"]["dataset"] == "BRETIGNY":
                            monte_dict[child_dir[0]]["params"]["dataset"] = "BRET"
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
        self.params_label = QLabel(str(self.params))
        df = pd.DataFrame()
        for img in self.image_paths.values():
            df = pd.concat([df, pd.DataFrame(img['params'], index=[0])], ignore_index=True)
        df.sort_values(by=['dataset', 'model', 'dtype', 'library', 'dataset_mode', 'dataset_method', 'balance'])
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

        rb2 = QRadioButton("real", self)
        rb2.toggled.connect(lambda: self.update_image("dtype", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
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
        rb1 = QRadioButton("cvnn")
        rb1.toggled.connect(lambda: self.update_image("library", rb1.text()))

        rb2 = QRadioButton("tensorflow", self)
        rb2.toggled.connect(lambda: self.update_image("library", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
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

    # def add_radiobuttons(self, name: str, options: List[str]):
    #     vlayout = QHBoxLayout()
    #     radio_buttons = []
    #     for option in options:
    #         rb = QRadioButton(option, self)
    #         rb.toggled.connect(lambda: self.update_image(name, rb.text()))
    #         radio_buttons.append(rb)
    #     self.btngroup.append(QButtonGroup())
    #     for button in radio_buttons:
    #         self.btngroup[-1].addButton(button)
    #     radio_buttons[0].setChecked(True)
    #     for button in radio_buttons:
    #         vlayout.addWidget(button)
    #     return vlayout

    def get_image(self, image_path: str):
        if image_path == '':
            pixmap = QPixmap(BASE_PATHS[self.params["dataset"]])
        else:
            pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1000, 700, QtCore.Qt.KeepAspectRatio)
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
