import sys
import os
from pathlib import Path
import random
import json
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from seaborn import heatmap
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QRadioButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QButtonGroup, QTableView, QHeaderView, QSizePolicy, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from typing import List
from results_reader import ResultReader, MonteCarloPlotter

from pdb import set_trace

if os.path.exists("/media/barrachina/data"):
    root_drive = Path("/media/barrachina/data")
elif os.path.exists("D:/"):
    root_drive = Path("D:/")
else:
    raise FileNotFoundError("Results path not found")

BASE_PATHS = {
    "BRET": str(root_drive / "datasets/PolSar/Bretigny-ONERA/bret-2003.png"),
    "OBER": str(
        root_drive / "datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6/PauliRGB_T1.bmp"),
    "SF-AIRSAR": str(root_drive / "datasets/PolSar/San Francisco/PolSF/SF-AIRSAR/SF-AIRSAR-Pauli.bmp"),
    "SF-RS2": str(root_drive / "datasets/PolSar/San Francisco/PolSF/SF-RS2/SF-RS2-Pauli.bmp"),
    "FLEVOLAND": str(root_drive / "datasets/PolSar/Flevoland/AIRSAR_Flevoland/T3/PauliRGB.bmp"),
    "GARON": str(root_drive / "/datasets/PolSar/garon/20141125-1_djit_rad.png")
}
GROUND_TRUTH_PATHS = {
    "BRET": str(root_drive / "datasets/PolSar/Bretigny-ONERA/labels_4roi.png"),
    "OBER": str(root_drive / "datasets/PolSar/Oberpfaffenhofen/ground_truth.png"),
    "SF-AIRSAR": str(root_drive / "datasets/PolSar/San Francisco/PolSF/SF-AIRSAR/SF-AIRSAR-label3d.png"),
    "SF-RS2": str(root_drive / "datasets/PolSar/San Francisco/PolSF/SF-RS2/SF-RS2-label3d.png"),
    "FLEVOLAND": str(root_drive / "datasets/PolSar/Flevoland/AIRSAR_Flevoland/ground_truth.png"),
    "GARON": str(root_drive / "/datasets/PolSar/garon/20141125-1_djit_rad-labels.png")
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
        if not index.isValid() or not (0 <= index.row() < self.rowCount() and 0 <= index.column() < self.columnCount()):
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
        self.ground_truth_image = QLabel()
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
        self.update_information()

    # Layouts
    def _get_accuracy_layout(self):
        self.acc_values_title = QLabel("Accuracy")
        myFont = QFont()
        myFont.setBold(True)
        self.acc_values_title.setFont(myFont)
        key = []
        key.append("Train OA")
        key.append("Train AA")
        key.append("Validation OA")
        key.append("Validation AA")
        key.append("Test OA")
        key.append("Test AA")
        self.tableAccWidget = QTableWidget()
        self.tableAccWidget.setRowCount(4)
        self.tableAccWidget.setColumnCount(6)
        self.tableAccWidget.setHorizontalHeaderLabels(key)
        self.tableAccWidget.setVerticalHeaderLabels(['mean', 'median', 'IQR', 'range'])
        self.tableAccWidget.setMinimumHeight(150)
        vbox = QVBoxLayout()
        vbox.addWidget(self.acc_values_title)
        vbox.addWidget(self.tableAccWidget)
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
        outer_vlay = QVBoxLayout()
        hl = QHBoxLayout()
        vlayout = QVBoxLayout()

        vlayout.addLayout(self.add_title(self.dataset_radiobutton(), name="Dataset"))
        vlayout.addLayout(self.add_title(self.model_radiobutton(), "Model"))
        vlayout.addLayout(self.add_title(self.dtype_radiobuttons(), "Dtype"))
        vlayout.addLayout(self.add_title(self.library_radiobutton(), "Library"))
        vlayout.addLayout(self.add_title(self.dataset_mode_radiobuttons(), "Dataset Mode"))
        vlayout.addLayout(self.add_title(self.model_method_radiobutton(), "Dataset Method"))
        vlayout.addLayout(self.add_title(self.balance_radiobuttons(), "Balance"))
        vlayout.addLayout(self.add_title(self.equiv_technique_radiobutton(), "Equivalent Technique"))

        hl.addLayout(vlayout)
        hl.addWidget(self.ground_truth_image)

        outer_vlay.addLayout(hl)
        outer_vlay.addStretch()
        outer_vlay.addLayout(self._get_accuracy_layout())
        return outer_vlay

    def dataset_mode_radiobuttons(self):
        vlayout = QHBoxLayout()
        self.coh_rb = QRadioButton("coh")
        self.coh_rb.toggled.connect(lambda: self.update_information("dataset_mode", self.coh_rb.text()))

        self.pauli_rb = QRadioButton("k", self)
        self.pauli_rb.toggled.connect(lambda: self.update_information("dataset_mode", self.pauli_rb.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(self.pauli_rb)
        self.btngroup[-1].addButton(self.coh_rb)

        self.coh_rb.setChecked(True)
        vlayout.addWidget(self.coh_rb)
        vlayout.addWidget(self.pauli_rb)
        vlayout.addStretch()

        return vlayout

    def dtype_radiobuttons(self):
        vlayout = QHBoxLayout()
        self.complex_dtype_rb = QRadioButton("complex")
        self.complex_dtype_rb.toggled.connect(
            lambda state: state and self.update_information("dtype", self.complex_dtype_rb.text()))

        self.real_dtype_rb = QRadioButton("real_imag", self)
        self.real_dtype_rb.toggled.connect(
            lambda state: state and self.update_information("dtype", self.real_dtype_rb.text()))

        rb2 = QRadioButton("amplitude_phase")
        rb2.toggled.connect(lambda state: state and self.update_information("dtype", rb2.text()))
        rb3 = QRadioButton("amplitude_only")
        rb3.toggled.connect(lambda state: state and self.update_information("dtype", rb3.text()))
        rb4 = QRadioButton("real_only")
        rb4.toggled.connect(lambda state: state and self.update_information("dtype", rb4.text()))

        self.btngroup.append(QButtonGroup())
        # self.btngroup[-1].addButton(self.real_dtype_rb)
        self.btngroup[-1].addButton(self.complex_dtype_rb)
        self.btngroup[-1].addButton(self.real_dtype_rb)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb4)

        self.complex_dtype_rb.setChecked(True)
        vlayout.addWidget(self.complex_dtype_rb)
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

    def equiv_technique_radiobutton(self):
        vlayout = QHBoxLayout()
        rb1 = QRadioButton("ratio_tp")
        rb1.toggled.connect(lambda: self.update_information("equiv_technique", rb1.text()))

        rb2 = QRadioButton("np", self)
        rb2.toggled.connect(lambda: self.update_information("equiv_technique", rb2.text()))

        rb3 = QRadioButton("alternate_tp")
        rb3.toggled.connect(lambda: self.update_information("equiv_technique", rb3.text()))

        rb4 = QRadioButton("none")
        rb4.toggled.connect(lambda: self.update_information("equiv_technique", rb4.text()))

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

        rb5 = QRadioButton("tan", self)
        rb5.toggled.connect(lambda: self.update_information("model", rb5.text()))

        rb6 = QRadioButton("cnn", self)
        rb6.toggled.connect(lambda: self.update_information("model", rb6.text()))

        rb7 = QRadioButton("mlp", self)
        rb7.toggled.connect(lambda: self.update_information("model", rb7.text()))

        rb8 = QRadioButton("expanded-mlp", self)
        rb8.toggled.connect(lambda: self.update_information("model", rb8.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb8)
        self.btngroup[-1].addButton(rb7)
        self.btngroup[-1].addButton(rb6)
        self.btngroup[-1].addButton(rb5)
        self.btngroup[-1].addButton(rb4)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addWidget(rb4)
        vlayout.addWidget(rb5)
        vlayout.addWidget(rb6)
        vlayout.addWidget(rb7)
        vlayout.addWidget(rb8)
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

        rb5 = QRadioButton("FLEVOLAND", self)
        rb5.toggled.connect(lambda: self.update_information("dataset", rb5.text()))

        self.btngroup.append(QButtonGroup())
        self.btngroup[-1].addButton(rb5)
        self.btngroup[-1].addButton(rb4)
        self.btngroup[-1].addButton(rb3)
        self.btngroup[-1].addButton(rb2)
        self.btngroup[-1].addButton(rb1)

        rb1.setChecked(True)
        vlayout.addWidget(rb1)
        vlayout.addWidget(rb2)
        vlayout.addWidget(rb3)
        vlayout.addWidget(rb4)
        vlayout.addWidget(rb5)
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

    # Update information
    def get_image(self, image_path: List[str]):
        if not image_path:
            pixmap = QPixmap(BASE_PATHS[self.params["dataset"]])
        else:
            image_path = random.choice(image_path)
            pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(900, 700, QtCore.Qt.KeepAspectRatio)
        self.label_image.setPixmap(scaled_pixmap)
        self.label_image.resize(scaled_pixmap.width(), scaled_pixmap.height())

    def get_image_ground_truth(self):
        pixmap = QPixmap(GROUND_TRUTH_PATHS[self.params["dataset"]])
        scaled_pixmap = pixmap.scaled(400, 200, QtCore.Qt.KeepAspectRatio)
        self.ground_truth_image.setPixmap(scaled_pixmap)
        self.ground_truth_image.resize(scaled_pixmap.width(), scaled_pixmap.height())

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

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

    def clear_conf_matrix(self):
        if hasattr(self, "conf_figure"):
            self.conf_figure.clear()
            self.conf_canvas.draw()

    def plot_conf_matrix(self, conf_list):
        if hasattr(self, "conf_figure"):
            self.conf_figure.clear()
            ax1 = self.conf_figure.add_subplot(131)
            ax2 = self.conf_figure.add_subplot(132)
            ax3 = self.conf_figure.add_subplot(133)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            heatmap(conf_list[0].drop('Total', axis=1).drop('Total'), ax=ax1, annot=True, cmap="rocket_r")
            heatmap(conf_list[1].drop('Total', axis=1).drop('Total'), ax=ax2, annot=True, cmap="rocket_r")
            if len(conf_list) > 2:
                heatmap(conf_list[2].drop('Total', axis=1).drop('Total'), ax=ax3, annot=True, cmap="rocket_r")
            self.conf_canvas.draw()

    def print_values(self, json_key):
        if json_key is not None and hasattr(self, "tableAccWidget"):
            self.acc_values_title.setText(f"Accuracy (total count {int(self.simulation_results.get_total_count(json_key))})")
            for i, stat in enumerate(["mean", "median", "iqr", 'range']):
                for j, dat in enumerate(["train", "val", "test"]):
                    for k, variable in enumerate(["accuracy", "average_accuracy"]):
                        self.tableAccWidget.setItem(i, 2*j + k, QTableWidgetItem(
                            self.simulation_results.get_eval_stat_string(json_key=json_key, dataset=dat,
                                                                         stat=stat, variable=variable)))
            header = self.tableAccWidget.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            # header.setMinimumSectionSize(200)
        elif hasattr(self, "tableAccWidget"):
            self.acc_values_title.setText("Accuracy")
            for i in range(len(["mean", "median", "iqr", 'range'])):
                self.tableAccWidget.setItem(i, 0, QTableWidgetItem(f"00.00%"))
                self.tableAccWidget.setItem(i, 1, QTableWidgetItem(f"00.00%"))
                self.tableAccWidget.setItem(i, 2, QTableWidgetItem(f"00.00%"))
                self.tableAccWidget.setItem(i, 3, QTableWidgetItem(f"00.00%"))
                self.tableAccWidget.setItem(i, 4, QTableWidgetItem(f"00.00%"))
                self.tableAccWidget.setItem(i, 5, QTableWidgetItem(f"00.00%"))

    def _verify_combinations(self, key, value):
        if value == "complex":
            if hasattr(self, 'cvnn_library_rb'):  # It wont exists if I still didnt create the radiobutton.
                self.cvnn_library_rb.setChecked(True)  # Set library cvnn
        elif value == "tensorflow":
            if hasattr(self, 'real_dtype_rb'):
                if self.complex_dtype_rb.isChecked():
                    self.real_dtype_rb.setChecked(True)  # Set real dtype
        elif key == 'dataset':
            if value == "OBER":
                if hasattr(self, 'coh_rb'):
                    self.coh_rb.setChecked(True)  # Set real dtype
                    self.coh_rb.setEnabled(False)
                if hasattr(self, 'pauli_rb'):
                    self.pauli_rb.setEnabled(False)
            else:
                if hasattr(self, 'coh_rb'):
                    self.coh_rb.setEnabled(True)
                if hasattr(self, 'pauli_rb'):
                    self.pauli_rb.setEnabled(True)
        elif key == "equiv_technique" and value != "ratio_tp":
            if hasattr(self, 'real_dtype_rb'):
                self.real_dtype_rb.setChecked(True)  # Set real dtype

    def update_information(self, key=None, value=None):
        if key and value:
            self.params[key] = value
            self._verify_combinations(key, value)
        self.params_label.setText(str(self.params))
        # Not yet working.
        # Try https://stackoverflow.com/questions/49929668/disable-and-enable-radiobuttons-from-another-radiobutton-in-pyqt4-python
        json_key = json.dumps(self.params, sort_keys=True)
        self.get_image(self.simulation_results.get_image(json_key))
        self.get_image_ground_truth()
        if self.simulation_results.data_exists(json_key):
            stats = self.simulation_results.get_stats(json_key=json_key)
            conf_stats = self.simulation_results.get_conf_stats(json_key=json_key)
            self.plot(stats)
            self.print_values(json_key)
            self.plot_conf_matrix(conf_stats)
        else:
            self.print_values(None)
            self.clear_plot()
            self.clear_conf_matrix()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
