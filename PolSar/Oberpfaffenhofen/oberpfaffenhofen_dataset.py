import scipy.io
import os
from os import path
import sys

if path.exists('/home/barrachina/Documents/onera/PolSar/'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar/')
    NOTIFY = False
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\onera\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\onera\PolSar')
    NOTIFY = False
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar/'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar/')
    NOTIFY = True
elif path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
    NOTIFY = False
else:
    raise FileNotFoundError("path of the dataset reader not found")
from dataset_reader import PolsarDatasetHandler

if os.path.exists('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen'):
    labels_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    t_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar\Oberpfaffenhofen'):
    labels_path = 'W:\HardDiskDrive\Documentos\GitHub\/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    t_path = 'W:\HardDiskDrive\Documentos\GitHub\datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = 'W:\HardDiskDrive\Documentos\GitHub\datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'):
    labels_path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    t_path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
elif path.exists("/home/cfren/Documents/onera/PolSar/Oberpfaffenhofen"):
    labels_path = '/home/cfren/Documents/data/PolSAR/Oberpfaffenhofen/Label_Germany.mat'
    t_path = '/home/cfren/Documents/data/PolSAR/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = '/home/cfren/Documents/data/PolSAR/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
else:
    raise FileNotFoundError("No path found for the requested dataset")


class OberpfaffenhofenDataset(PolsarDatasetHandler):

    def __init__(self, *args, **kwargs):
        super(OberpfaffenhofenDataset, self).__init__(name="OBER", mode="t", *args, **kwargs)

    def print_ground_truth(self, t=None, *args, **kwargs):
        if t is None:
            t = self.image
        super(OberpfaffenhofenDataset, self).print_ground_truth(t=t, *args, **kwargs)

    def open_image(self):
        labels = scipy.io.loadmat(labels_path)['label']
        image = self.open_t_dataset_t3(t_path)
        return image, self.sparse_to_categorical_2D(labels), labels
