import numpy as np
from imageio import imread
from pathlib import Path
from os import path
from typing import Tuple
from pdb import set_trace
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
if path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
    root_path = "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF"
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar')
elif path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
    root_path = "/home/cfren/Documents/onera/PolSar/San Francisco/PolSF"
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from dataset_reader import labels_to_rgb, SF_COLORS, PolsarDatasetHandler


AVAILABLE_IMAGES = {
    "SF-AIRSAR": {"x1": 0, "y1": 0, "x2": 1024, "y2": 900, "y_inverse": False},
    "SF-ALOS2": {"x1": 736, "y1": 2832, "x2": 3520, "y2": 7888, "y_inverse": True},
    # "SF-GF3": {"x1": 1144, "y1": 3464, "x2": 3448, "y2": 6376, "y_inverse": True},
    "SF-RS2": {"x1": 661, "y1": 7326, "x2": 2041, "y2": 9126, "y_inverse": False},
    # "SF-RISAT": {"x1": 2486, "y1": 4257, "x2": 7414, "y2": 10648, "y_inverse": False},   # RISAT is not Pol
}


class SanFranciscoDataset(PolsarDatasetHandler):

    def __init__(self, dataset_name: str, mode: str, *args, **kwargs):
        dataset_name = dataset_name.upper()
        assert dataset_name in AVAILABLE_IMAGES, f"Unknown data {dataset_name}."
        super(SanFranciscoDataset, self).__init__(name=dataset_name, mode=mode, *args, **kwargs)

    def print_ground_truth(self, t=None, *args, **kwargs):
        if t is None:
            t = self.image if self.mode == "t" else None
        super(SanFranciscoDataset, self).print_ground_truth(t=t, *args, **kwargs)

    def _get_labels(self):
        labels = imread(Path(root_path) / self.name / (self.name + "-label2d.png"))
        return labels

    def open_image(self, save_image: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        folder = "SAN_FRANCISCO_" + self.name[3:]
        labels = self._get_labels()
        one_hot_labels = self.sparse_to_categorical_2D(labels)
        if self.mode == "s":
            data = self.open_s_dataset(str(Path(root_path) / self.name / folder))
        elif self.mode == "t":
            data = self.open_t_dataset_t3(str(Path(root_path) / self.name / folder / "T4"))
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        data = data[
               AVAILABLE_IMAGES[self.name]["y1"]:AVAILABLE_IMAGES[self.name]["y2"],
               AVAILABLE_IMAGES[self.name]["x1"]:AVAILABLE_IMAGES[self.name]["x2"]
               ]
        assert data.shape[:-1] == one_hot_labels.shape[
                                  :-1], f"dataset of shape {data.shape[:-1]} not corresponding with " \
                                        f"labels of shape {one_hot_labels.shape[:-1]} for {self.name}"
        if AVAILABLE_IMAGES[self.name]["y_inverse"]:
            data = np.flip(data, axis=0)
        return data, one_hot_labels, labels
