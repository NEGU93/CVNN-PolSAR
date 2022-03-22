import sys
import os
import scipy.io
import numpy as np
import datetime as dt
from pathlib import Path
from pdb import set_trace
from typing import Optional
sys.path.insert(1, '../')
if os.path.exists("/home/barrachina/Documents/onera/PolSar"):
    sys.path.insert(1, "/home/barrachina/Documents/onera/PolSar")
    dataset_path = "/media/barrachina/data/datasets/PolSar/garon/polsar_mat"
    labels_path = "/media/barrachina/data/datasets/PolSar/garon/polsar_mat"
    NOTIFY = False
elif os.path.exists("/usr/users/gpu-prof/gpu_barrachina/onera/PolSar"):
    sys.path.insert(1, "/usr/users/gpu-prof/gpu_barrachina/onera/PolSar")
    dataset_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/garon/polsar_mat"
    labels_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/garon/polsar_mat"
    NOTIFY = True
elif os.path.exists("/scratchm/jbarrach/Garon"):
    sys.path.insert(1, "/scratchm/jbarrach/onera/PolSar")
    labels_path = "/scratchm/jbarrach/garon/polsar_mat"
    dataset_path = "/scratchm/jbarrach/garon/polsar_mat"
    NOTIFY = True
else:
    raise FileNotFoundError("path of the flevoland dataset not found")
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
        assert image_number < len(available_images)
        super(GaronDataset, self).__init__(root_path=dataset_path, name="GARON", mode=mode,
                                           *args, **kwargs)

    def get_image(self, image_number: Optional[int] = None) -> np.ndarray:
        if image_number is None:
            image_number = self.image_number
        if self.mode == 's':
            return np.load(self.root_path / available_images[image_number])
        else:
            raise ValueError(f"Sorry, dataset mode {self.mode} not supported")

    def get_dataset_date(self, image_number: Optional[int] = None):
        # TODO: parse the date?
        if image_number is None:
            image_number = self.image_number
        return dt.datetime.strptime(available_images[image_number][:8], "%m%d%Y").date()

    def get_sparse_labels(self) -> np.ndarray:
        return self.get_image().astype(int)


if __name__ == "__main__":
    save_to_mat(1)
    save_to_mat(2)
    save_to_mat(3)
    save_to_mat(4)
