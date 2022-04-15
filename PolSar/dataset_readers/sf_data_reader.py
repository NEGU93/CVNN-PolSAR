import numpy as np
from imageio import imread
from pathlib import Path
from os import path
from pdb import set_trace
import sys
sys.path.insert(1, '../')
if path.exists('/media/barrachina/data/datasets/PolSar/San Francisco/PolSF'):
    root_path = "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF"
elif path.exists('D:/datasets/PolSar/San Francisco/PolSF'):
    root_path = "D:/datasets/PolSar/San Francisco/PolSF"
elif path.exists("/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/San Francisco/PolSF"):
    root_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/San Francisco/PolSF"
elif path.exists('/home/cfren/Documents/onera/PolSar/San Francisco/PolSF'):
    root_path = "/home/cfren/Documents/onera/PolSar/San Francisco/PolSF"
elif path.exists('/scratchm/jbarrach'):
    root_path = None    # TODO
else:
    raise FileNotFoundError("path of the san francisco dataset not found")
from dataset_reader import PolsarDatasetHandler


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
        super(SanFranciscoDataset, self).__init__(root_path=str(Path(root_path) / dataset_name),
                                                  name=dataset_name, mode=mode, *args, **kwargs)
        self.orientation = "vertical"

    def get_sparse_labels(self):
        if root_path is None:
            raise FileNotFoundError("path of the san francisco dataset not found")
        labels = imread(Path(root_path) / self.name / (self.name + "-label2d.png"))
        return labels

    def get_image(self, save_image: bool = False) -> np.ndarray:
        if root_path is None:
            raise FileNotFoundError("path of the san francisco dataset not found")
        folder = "SAN_FRANCISCO_" + self.name[3:]
        if self.mode == "s":
            data = self.open_s_dataset(str(Path(root_path) / self.name / folder / "S2"))
        elif self.mode == "t":
            data = self.open_t_dataset_t3(str(Path(root_path) / self.name / folder / "T3"))
        elif self.mode == "k":
            mat = self.open_s_dataset(str(Path(root_path) / self.name / folder / "S2"))    # s11, s12, s22
            data = self._get_k_vector(HH=mat[:, :, 0], VV=mat[:, :, 2], HV=mat[:, :, 1])
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        data = data[
               AVAILABLE_IMAGES[self.name]["y1"]:AVAILABLE_IMAGES[self.name]["y2"],
               AVAILABLE_IMAGES[self.name]["x1"]:AVAILABLE_IMAGES[self.name]["x2"]
               ]
        if AVAILABLE_IMAGES[self.name]["y_inverse"]:
            data = np.flip(data, axis=0)
        return data


if __name__ == "__main__":
    # test_coh_matrix_generator(kernel_shape=1)
    data_handler = SanFranciscoDataset(mode='t', dataset_name="SF-AIRSAR")
    data_handler.print_image_png()
