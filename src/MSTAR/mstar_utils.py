import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pdb import set_trace
import os
from sklearn import preprocessing
import matplotlib.image as mpimg

root_path = Path('/media/barrachina/data/datasets/MSTAR/')


def plot_data(data):
    data = np.abs(data)
    arr = np.asarray(data)
    plt.imshow(arr, cmap='gray')
    plt.show()


def plot_jpg(mstar_data):
    fullfilename = str(root_path) + mstar_data[0]['path'] + '/' + mstar_data[0]['Filename'].upper()
    jpg_path = os.path.splitext(str(fullfilename))[0] + '.JPG'
    img = mpimg.imread(jpg_path)
    imgplot = plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    mstar_data = np.load(root_path/'data.npy', allow_pickle=True)
    plot_data(mstar_data[0]['data'])
    plot_jpg(mstar_data)


