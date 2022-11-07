import logging
import scipy.io
import sys
import os
from pathlib import Path
from pdb import set_trace
sys.path.insert(1, '../')
if os.path.exists('/home/barrachina/Documents/onera/src'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/src')
    dataset_path = "/media/barrachina/data/datasets/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = '/media/barrachina/data/datasets/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat'
    NOTIFY = False
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
    dataset_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat"
    NOTIFY = True
elif os.path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
    dataset_path = "/home/cfren/Documents/onera/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = "/home/cfren/Documents/onera/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat"
    NOTIFY = True
elif os.path.exists("/scratchm/jbarrach/Flevoland"):
    sys.path.insert(1, '/scratchm/jbarrach/onera/PolSar')
    labels_path = '/scratchm/jbarrach/Flevoland/Label_Flevoland_15cls.mat'
    dataset_path = '/scratchm/jbarrach/Flevoland/T3'
    NOTIFY = True
else:
    raise FileNotFoundError("path of the flevoland dataset not found")
from dataset_reader import PolsarDatasetHandler


class FlevolandDataset(PolsarDatasetHandler):

    def __init__(self, mode='t', *args, **kwargs):
        if mode != 't':
            logging.warning("Flevoland dataset must be coherency matrix. Mode parameter will be ignored.")
        super(FlevolandDataset, self).__init__(root_path=os.path.dirname(labels_path),
                                               name="FLEVOLAND", mode='t', *args, **kwargs)
        self.azimuth = "horizontal"

    @PolsarDatasetHandler.mode.setter
    def mode(self, mode):
        if mode.lower() != 't':
            raise ValueError(f"Cannot set mode on Flevoland data as it only supports coherency matrix (mode = 't')")

    def get_image(self):
        return self.open_t_dataset_t3(dataset_path)

    def get_sparse_labels(self):
        return scipy.io.loadmat(labels_path)['label']


if __name__ == '__main__':
    flev_handle = FlevolandDataset()
    img = flev_handle.get_image()
    labels = flev_handle.get_sparse_labels()
    flev_handle.print_ground_truth()







