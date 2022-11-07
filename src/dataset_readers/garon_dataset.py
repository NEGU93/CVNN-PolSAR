import sys
import os
import h5py
import pandas as pd
import scipy.io
import numpy as np
import datetime as dt
from pathlib import Path
from pdb import set_trace
from typing import Optional
sys.path.insert(1, '../')
if os.path.exists("/home/barrachina/Documents/onera/src"):
    sys.path.insert(1, "/home/barrachina/Documents/onera/src")
    dataset_path = "/media/barrachina/data/datasets/PolSar/garon/polsar_mat"
    labels_path = "/media/barrachina/data/datasets/PolSar/garon/labels"
    NOTIFY = False
elif os.path.exists("/usr/users/gpu-prof/gpu_barrachina/onera/PolSar"):
    sys.path.insert(1, "/usr/users/gpu-prof/gpu_barrachina/onera/PolSar")
    dataset_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/garon/polsar_mat"
    labels_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/garon/labels"
    NOTIFY = True
elif os.path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
    dataset_path = "/home/cfren/Documents/onera/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = "/home/cfren/Documents/onera/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat"
    NOTIFY = True
elif os.path.exists("/scratchm/jbarrach/Garon"):
    sys.path.insert(1, "/scratchm/jbarrach/onera/PolSar")
    labels_path = "/scratchm/jbarrach/garon/polsar_mat"
    dataset_path = "/scratchm/jbarrach/garon/labels"
    NOTIFY = True
else:
    raise FileNotFoundError("path of the Garon dataset not found")
from dataset_reader import PolsarDatasetHandler

available_images = {
    1: "20141125-1_djit_rad.npy",
    2: "20141201-1_djit_rad.npy",
    3: "20141203-1_djit_rad.npy",
    4: "20141212b-1_djit_rad.npy"
}


def save_to_mat(image_number: Optional[int]):
    np_mat = np.load(str(Path(dataset_path) / available_images[image_number]))  # HH, HV, VH, VV
    # mat_mat = scipy.io.loadmat('/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data/bretigny_seg.mat')
    # set_trace()
    to_save = {
        "HH": np_mat[:, :, 0], "HV": np_mat[:, :, 1], "VV": np_mat[:, :, 3],
        "__header__": f"Python, Platform: {sys.platform}, Created on: {dt.date.today().strftime('%a %b %d %H:%M:%S %Y')}",
        "date": dt.date.today().strftime("%d-%b-%Y"),
        "__version__": "1.0", "__globals__": [], "classtype": "PolarImage"
    }
    scipy.io.savemat(file_name=(Path(dataset_path) / available_images[image_number]).with_suffix(".mat"), mdict=to_save)


class GaronDataset(PolsarDatasetHandler):
    
    def __init__(self, mode: str, image_number: int = 1, *args, **kwargs):
        self.image_number = int(image_number)
        assert image_number < len(available_images) + 1
        super(GaronDataset, self).__init__(root_path=dataset_path, name="GARON", mode=mode,
                                           *args, **kwargs)
        self.azimuth = "vertical"

    def get_image(self, image_number: Optional[int] = None) -> np.ndarray:
        if image_number is None:
            image_number = self.image_number
        s_raw = np.load(self.root_path / available_images[image_number])
        if self.mode == 's':
            return s_raw[:, :, :-1]     # Don't send VH TODO: IS ORDER OK?
        elif self.mode == 't':
            return self.numpy_coh_matrix(HH=s_raw[:, :, 0], VV=s_raw[:, :, 3], HV=s_raw[:, :, 1], kernel_shape=1)
        else:
            raise ValueError(f"Sorry, dataset mode {self.mode} not supported")

    def get_dataset_date(self, image_number: Optional[int] = None):
        if image_number is None:
            image_number = self.image_number
        return dt.datetime.strptime(available_images[image_number][:8], "%Y%m%d").date()

    def get_sparse_labels(self, image_number: Optional[int] = None) -> np.ndarray:
        if image_number is None:
            image_number = self.image_number
        f = h5py.File(Path(labels_path) / (available_images[image_number][:-4] + "-labels.mat"), 'r')
        seg = np.array(f['labels'], dtype=int)
        seg[seg == 5] = 0       # Remove canal labels.
        return seg


if __name__ == "__main__":
    data_handler = GaronDataset(mode='s', image_number=1)
    data_handler.print_ground_truth(transparent_image=0.7, showfig=False,
                                    path=str(Path(labels_path) / (available_images[1][:-4] + "-labels.png")))
    # for key in [4]:
    #     data_handler = GaronDataset(mode='s', image_number=key)
    #     set_trace()
