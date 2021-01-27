import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pdb import set_trace
import os
from sklearn import preprocessing
from PIL import Image


def plot_data(data):
    data = np.abs(data)
    arr = np.asarray(data)
    plt.imshow(arr, cmap='gray', vmin=np.min(arr), vmax=np.max(arr))
    plt.show()


if __name__ == '__main__':
    root_path = Path('/media/barrachina/data/datasets/MSTAR/')
    mstar_data = np.load(root_path/'data.npy', allow_pickle=True)
    plot_data(mstar_data[0]['data'])
    fullfilename = str(root_path) + mstar_data[0]['path'] + '/' + mstar_data[0]['Filename'].upper()
    image = Image.open(os.path.splitext(str(fullfilename))[0] + '.JPG')
    image.show()

