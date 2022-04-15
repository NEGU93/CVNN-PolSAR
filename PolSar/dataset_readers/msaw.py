import os
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from skimage import io
# import solaris.preproc as sp
import cv2
import matplotlib.pyplot as plt
from dataset_reader import PolsarDatasetHandler
from pdb import set_trace

if os.path.exists('/media/barrachina/data/datasets/PolSar/MSAW'):
    labels_path = '/media/barrachina/data/datasets/PolSar/MSAW/SN6_buildings_AOI_11_Rotterdam_train/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv'
    t_path = '/media/barrachina/data/datasets/PolSar/MSAW/SAR-SLC'


class MSAWDataset(PolsarDatasetHandler):

    def __init__(self, *args, **kwargs):
        super(MSAWDataset, self).__init__(root_path=os.path.dirname(labels_path),
                                          name="MSAW", mode="t", *args, **kwargs)

    def open_tif(self, path):
        try:
            im = cv2.imread(str(path))
            if im is not None:
                print("OpenCV worked")
                return im
        except Exception as e:
            print(f"OpenCV failed with error {e}")
        finally:
            try:
                im = plt.imread(str(path))
                if im is not None:
                    print("Matplotlib worked")
                    return im
            except Exception as e:
                print(f"Matplotlib failed with error {e}")
            finally:
                try:
                    im = Image.open(str(path))
                    if im is not None:
                        print("PIL worked")
                        return im
                except Exception as e:
                    print(f"PIL failed with error {e}")
                finally:
                    try:
                        im = io.imread(str(path))
                        if im is not None:
                            print("skimage worked")
                            return im
                    except Exception as e:
                        print(f"skimage failed with error {e}")
                    finally:
                        print("Nothing works :(")
        return None

    def get_image(self):
        # TODO: This still dont work correctly because my method waits for the full image and not the tiles already.
        tiles = []
        for file in Path(t_path).glob('*.tif'):
            im = self.open_tif(file)
            set_trace()
            tiles.append(im)
        return np.array(tiles)

    def get_sparse_labels(self):
        pass


if __name__ == "__main__":
    print("First Test")
    MSAWDataset().get_dataset(method="random", size=128, stride=25, pad="same")