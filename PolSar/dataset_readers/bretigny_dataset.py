import scipy.io
import os
import numpy as np
import tensorflow as tf
from os import path
import sys

from dataset_reader import PolsarDatasetHandler

sys.path.insert(1, '../')

if os.path.exists('/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'):
    path = '/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('D:/datasets/PolSar/Bretigny-ONERA/data'):
    path = 'D:/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'):
    path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('/home/cfren/Documents/onera/PolSar/Bretigny-ONERA/data'):
    path = '/home/cfren/Documents/onera/PolSar/Bretigny-ONERA/data'
else:
    path = None


class BretignyDataset(PolsarDatasetHandler):

    def __init__(self, mode: str, balanced: bool = False, *args, **kwargs):
        self.balanced = balanced
        super(BretignyDataset, self).__init__(root_path=path, name="BRET", mode=mode, *args, **kwargs)

    def print_ground_truth(self, t=None, *args, **kwargs):
        if t is None:
            t = self.get_image() if self.mode == "t" else None
        super(BretignyDataset, self).print_ground_truth(t=t, *args, **kwargs)

    def get_image(self) -> np.ndarray:
        if self.mode == "s":
            return self._get_bret_s_dataset()
        elif self.mode == "t":
            return self._get_bret_coherency_dataset()
        elif self.mode == "k":
            return self._get_bret_k_dataset()
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

    def get_sparse_labels(self):
        if path is None:
            raise FileNotFoundError("Bretigny dataset path not found")
        if not self.balanced:
            seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI.mat')
        else:
            seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI_balanced.mat')
        seg['image'] = seg['image'][:-3]
        return seg['image']

    """
        PRIVATE
    """

    def _open_data(self):
        if path is None:
            raise FileNotFoundError("Bretigny dataset path not found")
        mat = scipy.io.loadmat(path + '/bretigny_seg.mat')
        mat['HH'] = mat['HH'][:-3]
        mat['HV'] = mat['HV'][:-3]
        mat['VV'] = mat['VV'][:-3]
        return mat

    def _get_bret_coherency_dataset(self):
        mat = self._open_data()
        T = self.numpy_coh_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'], kernel_shape=self.coh_kernel_size)
        return T

    def _get_bret_k_dataset(self):
        mat = self._open_data()
        k = self._get_k_vector(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
        return k

    def _get_bret_s_dataset(self):
        mat = self._open_data()
        s = np.array([mat['HH'], mat['VV'], mat['HV']])
        s = tf.transpose(s, perm=[1, 2, 0])
        return s


if __name__ == "__main__":
    data_handler = BretignyDataset(mode='t')
    data_handler.get_image()
    data_handler.print_image_png(savefile=True, showfig=True)
