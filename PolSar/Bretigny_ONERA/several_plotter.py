import plotly.express as px
from skimage.io import imread
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
from pdb import set_trace
from typing import Optional, Dict
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QSlider, QTableView
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import sys


params = {
    "dtype": 'complex',
    "model": '0',
    "library": 'cvnn',
    "dataset_mode": 'k',
    "dataset_split": 'random',
    "boxcar": '3',
    "loss": 'conventional',
    "labels": 'original'
}
keys = ['accuracy', 'val_accuracy']


def get_paths(root_dir: str = "/media/barrachina/data/results/Bretigny/after_icassp") -> dict:
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
                    if (Path(child_dir[0]) / 'history_dict').is_file():
                        try:
                            boxcar_index = simu_params.split().index('--boxcar')
                        except ValueError:
                            boxcar_index = -1
                        try:
                            model_index = simu_params.split().index('--model')
                        except ValueError:
                            model_index = -1
                        monte_dict[child_dir[0]] = {
                            "data": str(Path(child_dir[0]) / 'history_dict'),
                            "image": str(Path(child_dir[0]) / 'prediction.png'),
                            "params": {
                                "dtype": f"{'real' if 'real_mode' in simu_params else 'complex'}",
                                "model": f"{simu_params.split()[model_index + 1] if model_index != -1 else '0'}",
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset_split": f"{'sections' if 'split_datasets' in simu_params else 'random'}",
                                "boxcar": f"{simu_params.split()[boxcar_index + 1] if boxcar_index != -1 else '3'}",
                                "loss": f"{'conventional' if 'weighted_loss' not in simu_params else 'weighted'}",
                                "labels": f"{'balanced' if 'balanced' in child_dir[0] else 'original'}"
                            }
                        }
                    else:
                        print("No history_dict found on path " + child_dir[0])
            else:
                print("No model_summary.txt found in " + child_dir[0])
    return monte_dict


user_dict = get_paths("/media/barrachina/data/results/Bretigny/after_icassp")

"""
    Update menus dictionaries
"""


def _get_dataset_mode_dict(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'k',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_mode": "k"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'coh',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_mode": "coh"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_dtype(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'complex',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dtype": "complex"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'real',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dtype": "real"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_library(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'cvnn',
                'args': [
                    {'y': [
                        _update_plot(new_value={"library": "cvnn"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'tensorflow',
                'args': [
                    {'y': [
                        _update_plot(new_value={"library": "tensorflow"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_dataset_split(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'random',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_split": "random"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'sections',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_split": "sections"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_loss(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'conventional',
                'args': [
                    {'y': [
                        _update_plot(new_value={"loss": "conventional"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'weighted',
                'args': [
                    {'y': [
                        _update_plot(new_value={"loss": "weighted"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_boxcar(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'default',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "default"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "1"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '5',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "5"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '3',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "3"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_model(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'default',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "default"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '0',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "0"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "1"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '2',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "2"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '4',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "3"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "4"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


"""
    Other
"""


def _get_value(new_value: Optional[dict]):
    if new_value is not None:
        assert len(new_value) == 1
        new_key = next(iter(new_value))
        params[new_key] = new_value[new_key]
    for key in user_dict.keys():  # get one by one to see if I get it.
        coincidence = True
        for param_to_match, value_to_match in params.items():
            if user_dict[key]['params'][param_to_match] != value_to_match:
                coincidence = False
                break
        if coincidence:
            return user_dict[key]['data']
    print("WARNING: Match not found, printing default")
    return user_dict[list(user_dict)[0]]['data']


def _update_plot(new_value: Optional[dict]):
    history_path = _get_value(new_value)
    with open(history_path, 'rb') as f:
        saved_history = pickle.load(f)
    df = pd.DataFrame(saved_history)[keys]
    return df


def live_plot():
    df = _update_plot(new_value=None)
    plot_lines = []
    for col in df.columns:
        plot_lines.append(go.Scatter(
            x=df.index, y=df[col], name=col, mode='lines'
        ))
    updatemenus = [
        _get_dataset_mode_dict(),
        _get_dataset_dtype(0.1),
        _get_dataset_library(0.2),
        _get_dataset_dataset_split(0.3),
        _get_dataset_loss(0.4),
        _get_dataset_boxcar(0.5),
        _get_dataset_model(0.6)
    ]

    layout = go.Layout(
        updatemenus=updatemenus
    )
    fig = go.Figure(data=plot_lines, layout=layout)
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
        self.params = params
        self.params_label = QLabel(str(self.params))
        df = pd.DataFrame()
        for img in self.image_paths.values():
            df = pd.concat([df, pd.DataFrame(img['params'], index=[0])], ignore_index=True)
        df.sort_values(by=['labels', 'dtype', 'dataset_mode', 'dataset_split', 'loss', 'boxcar'])
        self.btngroup = []
        widget = QWidget()
        hlayout = QHBoxLayout()

        self.get_image('')
        hlayout.addLayout(self.radiobuttons())
        img_layout = QVBoxLayout()
        self.params_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.params_label)
        img_layout.addWidget(self.label_image)
        hlayout.addLayout(img_layout)

        outer_layout = QVBoxLayout()
        outer_layout.addLayout(hlayout)
        self.tableView = QTableView()
        # self.verticalLayout.addWidget(self.tableView)
        self.tableView.setModel(DataFrameModel(df))
        outer_layout.addWidget(self.tableView)
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
        self.show()

    def radiobuttons(self):
        vlayout = QVBoxLayout()

        vlayout.addLayout(self.add_title(self.dtype_radiobuttons(), "Dtype"))
        vlayout.addLayout(self.add_title(self.library_radiobuttons(), "Library"))
        vlayout.addLayout(self.add_title(self.dataset_mode_radiobuttons(), "Dataset Mode"))
        vlayout.addLayout(self.add_title(self.dataset_split_radiobuttons(), "Dataset Split"))
        vlayout.addLayout(self.add_title(self.loss_radiobuttons(), "Loss"))
        vlayout.addLayout(self.add_title(self.labels_radiobuttons(), "Loss"))
        vlayout.addLayout(self.model_slider())
        vlayout.addLayout(self.boxcar_slider())

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

    def dtype_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("complex", self)
        rb1.toggled.connect(lambda: self.update_image("dtype", rb1.text()))

        rb2 = QRadioButton("real", self)
        rb2.toggled.connect(lambda: self.update_image("dtype", rb2.text()))

        self.rb = rb2

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def library_radiobuttons(self):
        vlayout = QHBoxLayout()
        self.rb1 = QRadioButton("cvnn", self)
        self.rb1.toggled.connect(lambda: self.update_image("library", self.rb1.text()))

        rb2 = QRadioButton("tensorflow", self)
        rb2.toggled.connect(lambda: self.update_image("library", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(self.rb1)

        self.rb1.setChecked(True)
        vlayout.addWidget(self.rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def labels_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("original", self)
        rb1.toggled.connect(lambda: self.update_image("labels", rb1.text()))

        rb2 = QRadioButton("balanced", self)
        rb2.toggled.connect(lambda: self.update_image("labels", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def dataset_mode_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("k", self)
        rb1.toggled.connect(lambda: self.update_image("dataset_mode", rb1.text()))

        rb2 = QRadioButton("coh", self)
        rb2.toggled.connect(lambda: self.update_image("dataset_mode", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def dataset_split_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("random", self)
        rb1.toggled.connect(lambda: self.update_image("dataset_split", rb1.text()))

        rb2 = QRadioButton("sections", self)
        rb2.toggled.connect(lambda: self.update_image("dataset_split", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def loss_radiobuttons(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("conventional", self)
        rb1.toggled.connect(lambda: self.update_image("loss", rb1.text()))

        rb2 = QRadioButton("weighted", self)
        rb2.toggled.connect(lambda: self.update_image("loss", rb2.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)

        return vlayout

    def model_slider(self):
        layout = QVBoxLayout()
        l1 = QLabel("Model")
        l1.setAlignment(Qt.AlignCenter)
        layout.addWidget(l1)
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(0)
        sl.setMaximum(4)
        sl.setValue(0)
        sl.setTickInterval(1)
        sl.setTickPosition(QSlider.TicksBelow)
        sl.valueChanged[int].connect(lambda: self.update_image("model", str(sl.value())))
        layout.addWidget(sl)
        return layout

    def boxcar_slider(self):
        layout = QVBoxLayout()
        l1 = QLabel("Boxcar")
        l1.setAlignment(Qt.AlignCenter)
        layout.addWidget(l1)
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(1)
        sl.setMaximum(5)
        sl.setValue(3)
        sl.setTickInterval(2)
        sl.setTickPosition(QSlider.TicksBelow)
        sl.valueChanged[int].connect(lambda: self.update_image("boxcar", str(sl.value())))
        layout.addWidget(sl)
        return layout

    def get_image(self, image_path: str):
        if image_path == '':
            pixmap = QPixmap('/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/bret-2003.png')
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
                if key == "boxcar" and param["dataset_mode"] == 'k':
                    continue    # saved
                else:
                    return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

