import scipy.io
import os
from os import path
from pathlib import Path
import sys
sys.path.insert(1, '../')
from dataset_reader import PolsarDatasetHandler

if os.path.exists('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen'):
    labels_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    t_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
elif os.path.exists('D:/datasets/PolSar/Oberpfaffenhofen'):
    labels_path = 'D:/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat'
    t_path = 'D:/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = 'D:/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen'
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
elif path.exists("/scratchm/jbarrach/onera/PolSar/Oberpfaffenhofen"):
    labels_path = '/scratchm/jbarrach/Oberpfaffenhofen/Label_Germany.mat'
    t_path = '/scratchm/jbarrach/Oberpfaffenhofen//ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6'
    s_path = ''
else:
    raise FileNotFoundError("No path found for the requested dataset")


class OberpfaffenhofenDataset(PolsarDatasetHandler):

    def __init__(self, mode='t', *args, **kwargs):
        if mode != 't':
            print("WARNING: Flevoland dataset must be coherency matrix. mode parameter will be ignored")
        super(OberpfaffenhofenDataset, self).__init__(root_path=os.path.dirname(labels_path),
                                                      name="OBER", mode="t", *args, **kwargs)
        self.orientation = "vertical"

    def get_image(self):
        return self.open_t_dataset_t3(t_path)

    def get_sparse_labels(self):
        return scipy.io.loadmat(labels_path)['label']


if __name__ == "__main__":
    print("First Test")
    OberpfaffenhofenDataset().get_dataset(method="random", size=128, stride=25, pad="same")
    print("First one done")
    OberpfaffenhofenDataset(classification=True).get_dataset(method="random", size=12, stride=1, pad="same")
    print("Second one done")
    OberpfaffenhofenDataset(classification=True).get_dataset(method="random", size=1, stride=1, pad="same")
