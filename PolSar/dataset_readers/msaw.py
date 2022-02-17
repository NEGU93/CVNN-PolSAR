import os
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from skimage import io
import cv2
import matplotlib.pyplot as plt
from dataset_reader import PolsarDatasetHandler
from pdb import set_trace

if os.path.exists('/media/barrachina/data/datasets/PolSar/MSAW'):
    labels_path = '/media/barrachina/data/datasets/PolSar/MSAW/SN6_buildings_AOI_11_Rotterdam_train/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv'
    t_path = '/media/barrachina/data/datasets/PolSar/MSAW/SN6_buildings_AOI_11_Rotterdam_train/train/AOI_11_Rotterdam/SAR-Intensity'


class MSAWDataset(PolsarDatasetHandler):

    def __init__(self, *args, **kwargs):
        super(MSAWDataset, self).__init__(root_path=os.path.dirname(labels_path),
                                          name="MSAW", mode="t", *args, **kwargs)

    def print_ground_truth(self, t=None, *args, **kwargs):
        if t is None:
            t = self.get_image()
        super(MSAWDataset, self).print_ground_truth(t=t, path=Path(os.path.dirname(labels_path)) / "ground_truth.png",
                                                    *args, **kwargs)

    def get_image(self):
        # TODO: This still dont work correctly because my method waits for the full image and not the tiles already.
        tiles = []
        for file in Path(t_path).glob('*.tif'):
            # im = cv2.imread(str(file))
            # im = plt.imread(str(file))
            # im = Image.open(str(file))
            im = io.imread(str(file))
            set_trace()
            tiles.append(im)
        return np.array(tiles)

    def get_sparse_labels(self):
        pass


if __name__ == "__main__":
    print("First Test")
    MSAWDataset().get_dataset(method="random", size=128, stride=25, pad="same")