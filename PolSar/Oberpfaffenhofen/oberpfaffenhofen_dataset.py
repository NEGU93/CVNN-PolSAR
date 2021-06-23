import spectral.io.envi as envi
from pathlib import Path
import scipy.io
from pdb import set_trace
import os
from os import path
import sys

if path.exists('/home/barrachina/Documents/onera/PolSar/'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar/')
    NOTIFY = False
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar/'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar/')
    NOTIFY = True
else:
    raise FileNotFoundError("path of the dataset reader not found")
from dataset_reader import open_dataset_t6, get_dataset_for_segmentation, get_dataset_for_classification

if os.path.exists('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen'):
    labels = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/Oberpfaffenhofen/Label_Germany.mat'):
    labels = '/usr/users/gpu-prof/gpu_barrachina/Oberpfaffenhofen/Label_Germany.mat'
    path = '/usr/users/gpu-prof/gpu_barrachina/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
else:
    raise FileNotFoundError("No path found for the requested dataset")

def get_ober_dataset_for_segmentation(size: int = 128, stride: int = 25, debug=False):
    return get_dataset_for_segmentation(path, labels, size, stride, debug)


def get_ober_dataset_for_classification():
    return get_dataset_for_classification(path, labels)


def open_dataset_ober():
    """
    Opens the t6 dataset of Oberpfaffenhofen with the corresponding labels.
    :return: Tuple (T, labels)
        - T: Image as a numpy array of size hxwxB=1300x1200x21 where h and w are the height and width of the
            spatial dimensions respectively, B is the number of complex bands.
        - labels: numpy array of size 1300x1200 where each pixel has value:
            0: Unlabeled
            1: Built-up Area
            2: Wood Land
            3: Open Area
    """
    return open_dataset_t6(path, labels)


def open_dataset_s2():
    """
    Opens the s2 dataset of Oberpfaffenhofen with the corresponding labels.
    :return: Tuple (T, labels)
        - [s_11, s_12, s_21, s_22]: Image as a numpy array.
        - labels: numpy array.
    """
    path = Path('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen')
    labels = scipy.io.loadmat('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat')['label']

    # http://www.spectralpython.net/fileio.html#envi-headers
    s_11_meta = envi.open(path / 's11.bin.hdr', path / 's11.bin')
    s_12_meta = envi.open(path / 's12.bin.hdr', path / 's12.bin')
    s_21_meta = envi.open(path / 's21.bin.hdr', path / 's21.bin')
    s_22_meta = envi.open(path / 's22.bin.hdr', path / 's22.bin')

    s_11 = s_11_meta.read_band(0)
    s_12 = s_12_meta.read_band(0)
    s_21 = s_21_meta.read_band(0)
    s_22 = s_22_meta.read_band(0)

    return [s_11, s_12, s_21, s_22], labels
