import spectral.io.envi as envi
from pathlib import Path
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from pdb import set_trace
from typing import Tuple
from cvnn.utils import standarize, randomize


def separate_dataset(data, window_size: int = 9, stride: int = 3):
    """
    Gets each pixel of the dataset with some surrounding pixels.
    :param data:
    :param window_size:
    :param stride:
    :return:
    """
    assert window_size % 2 == 1, "Window size must be odd, got " + str(window_size)
    n_win = int((window_size - 1) / 2)
    rows = np.arange(n_win, data.shape[0] - n_win, stride)
    cols = np.arange(n_win, data.shape[1] - n_win, stride)
    result = np.empty((len(cols) * len(rows), window_size, window_size), dtype=data.dtype)
    k = 0
    for row in rows.astype(int):
        for col in cols.astype(int):
            result[k] = data[row - n_win:row + n_win + 1, col - n_win:col + n_win + 1]
            k += 1
    return result


def open_dataset_t6():
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
    labels = scipy.io.loadmat('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat')['label']
    path = Path(
        '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6')
    T = np.zeros(labels.shape + (21,), dtype=complex)

    T[:, :, 0] = standarize(envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0))
    T[:, :, 1] = standarize(envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0))
    T[:, :, 2] = standarize(envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0))
    T[:, :, 3] = standarize(envi.open(path / 'T44.bin.hdr', path / 'T44.bin').read_band(0))
    T[:, :, 4] = standarize(envi.open(path / 'T55.bin.hdr', path / 'T55.bin').read_band(0))
    T[:, :, 5] = standarize(envi.open(path / 'T66.bin.hdr', path / 'T66.bin').read_band(0))

    T[:, :, 6] = standarize(envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0))
    T[:, :, 7] = standarize(envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0))
    T[:, :, 8] = standarize(envi.open(path / 'T14_real.bin.hdr', path / 'T14_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T14_imag.bin.hdr', path / 'T14_imag.bin').read_band(0))
    T[:, :, 9] = standarize(envi.open(path / 'T15_real.bin.hdr', path / 'T15_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T15_imag.bin.hdr', path / 'T15_imag.bin').read_band(0))
    T[:, :, 10] = standarize(envi.open(path / 'T16_real.bin.hdr', path / 'T16_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T16_imag.bin.hdr', path / 'T16_imag.bin').read_band(0))

    T[:, :, 11] = standarize(envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0))
    T[:, :, 12] = standarize(envi.open(path / 'T24_real.bin.hdr', path / 'T24_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T24_imag.bin.hdr', path / 'T24_imag.bin').read_band(0))
    T[:, :, 13] = standarize(envi.open(path / 'T25_real.bin.hdr', path / 'T25_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T25_imag.bin.hdr', path / 'T25_imag.bin').read_band(0))
    T[:, :, 14] = standarize(envi.open(path / 'T26_real.bin.hdr', path / 'T26_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T26_imag.bin.hdr', path / 'T26_imag.bin').read_band(0))

    T[:, :, 15] = standarize(envi.open(path / 'T34_real.bin.hdr', path / 'T34_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T34_imag.bin.hdr', path / 'T34_imag.bin').read_band(0))
    T[:, :, 16] = standarize(envi.open(path / 'T35_real.bin.hdr', path / 'T35_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T35_imag.bin.hdr', path / 'T35_imag.bin').read_band(0))
    T[:, :, 17] = standarize(envi.open(path / 'T36_real.bin.hdr', path / 'T36_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T36_imag.bin.hdr', path / 'T36_imag.bin').read_band(0))

    T[:, :, 18] = standarize(envi.open(path / 'T45_real.bin.hdr', path / 'T45_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T45_imag.bin.hdr', path / 'T45_imag.bin').read_band(0))
    T[:, :, 19] = standarize(envi.open(path / 'T46_real.bin.hdr', path / 'T46_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T46_imag.bin.hdr', path / 'T46_imag.bin').read_band(0))

    T[:, :, 20] = standarize(envi.open(path / 'T56_real.bin.hdr', path / 'T56_real.bin').read_band(0) + \
                             1j * envi.open(path / 'T56_imag.bin.hdr', path / 'T56_imag.bin').read_band(0))
    return T, labels


def remove_unlabeled(x, y):
    """
    Removes the unlabeled pixels from both image and labels
    :param x: image input
    :param y: labels inputs. all values of 0's will be eliminated and its corresponging values of x
    :return: tuple (x, y) without the unlabeled pixels.
    """
    mask = y != 0
    return x[mask], y[mask]


def labels_to_ground_truth(labels, showfig=False, savefig=False) -> np.ndarray:
    """
    Transforms the labels to a RGB format so it can be drawn as images
    :param labels: The labels to be transformed to rgb
    :param showfig: boolean. If true it will show the generated ground truth image
    :param savefig: boolean. If true it will save the generated ground truth image
    :return: numpy array of the ground truth RGB image
    """
    colors = np.array([
        [1, 0.349, 0.392],      # Red; Built-up Area
        [0.086, 0.858, 0.576],  # Green; Wood Land
        [0.937, 0.917, 0.352]   # Yellow; Open Area
    ])
    ground_truth = np.zeros(labels.shape + (3,), dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                ground_truth[i, j] = colors[labels[i, j] - 1]
    plt.imshow(ground_truth)
    if showfig:
        plt.show()
    if savefig:
        plt.imsave("ground_truth.pdf", ground_truth)
        tikzplotlib.save("ground_truth.tex")
    return ground_truth


def sparse_to_categorical(labels) -> np.ndarray:
    ground_truth = np.zeros(labels.shape + (3,), dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                ground_truth[i, j, labels[i, j] - 1] = 1.
    return ground_truth


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


def separate_train_test_pixels(x, y, ratio=0.1):
    """
    Separates each pixel of the dataset in train and test set.
    The returned dataset will be randomized.
    :param x: dataset images
    :param y: dataset labels
    :param ratio: ratio of the train set, example, 0.8 will generate 80% of the dataset as train set and 20% as test set
    :return: Tuple x_train, y_train, x_test, y_test
    """
    classes = set(y)
    x_ordered_database = []
    y_ordered_database = []
    for cls in classes:
        mask = y == cls
        x_ordered_database.append(x[mask])
        y_ordered_database.append(y[mask])
    len_train = int(y.shape[0] * ratio / len(classes))
    x_train = x_ordered_database[0][:len_train]
    x_test = x_ordered_database[0][len_train:]
    y_train = y_ordered_database[0][:len_train]
    y_test = y_ordered_database[0][len_train:]
    for i in range(len(y_ordered_database)):
        assert (y_ordered_database[i] == i).all()
        assert len(y_ordered_database[i]) == len(x_ordered_database[i])
        if i != 0:
            x_train = np.concatenate((x_train, x_ordered_database[i][:len_train]))
            x_test = np.concatenate((x_test, x_ordered_database[i][len_train:]))
            y_train = np.concatenate((y_train, y_ordered_database[i][:len_train]))
            y_test = np.concatenate((y_test, y_ordered_database[i][len_train:]))
    x_train, y_train = randomize(x_train, y_train)
    x_test, y_test = randomize(x_test, y_test)
    return x_train, y_train, x_test, y_test


def get_dataset_for_classification():
    """
    Gets dataset ready to be processed for the mlp model.
    The dataset will be 2D dimensioned where the first element will be each pixel that will have 21 complex values each.
    :return: Tuple (T, labels)
    """
    T, labels = open_dataset_t6()
    labels_to_ground_truth(labels)
    T, labels = remove_unlabeled(T, labels)
    labels -= 1  # map [1, 3] to [0, 2]
    T = T.reshape(-1, T.shape[-1])  # Image to 1D
    labels = labels.reshape(np.prod(labels.shape))
    return T, labels


def sliding_window_operation(im, lab, size: int = 128, stride: int = 128, pad: int = 0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts many sub-images from one big image. Labels included.
    Using the Sliding Window Operation defined in:
        https://www.mdpi.com/2072-4292/10/12/1984
    :param im: Image dataset
    :param lab: pixel-wise labels dataset
    :param size: Size of the desired mini new images
    :param stride: stride between images, use stride=size for images not to overlap
    :param pad: Pad borders
    :return: tuple of numpy arrays (tiles, label_tiles)
    """
    tiles = []
    label_tiles = []
    assert im.shape[0] > size and im.shape[1] > size
    if pad:
        im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)))
        lab = np.pad(lab, ((pad, pad), (pad, pad)))
    for x in range(0, im.shape[0] - size, stride):
        for y in range(0, im.shape[1] - size, stride):
            slice_x = slice(x, x + size)
            slice_y = slice(y, y + size)
            tiles.append(im[slice_x, slice_y])
            label_tiles.append(lab[slice_x, slice_y])
    assert np.all([p.shape == (size, size, im.shape[2]) for p in tiles])
    assert np.all([p.shape == (size, size, lab.shape[2]) for p in label_tiles])
    if not pad:     # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
        assert int(np.shape(tiles)[0]) == int((np.floor((im.shape[0]-size)/stride)+1)*(np.floor((im.shape[1]-size)/stride)+1))
    return np.array(tiles), np.array(label_tiles)


def load_image_train(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def get_dataset_for_segmentation(size: int = 128, stride: int = 25, debug=False) -> tf.data.Dataset:
    T, labels = open_dataset_t6()
    labels_to_ground_truth(labels, showfig=debug)
    labels = sparse_to_categorical(labels)
    patches, label_patches = sliding_window_operation(T, labels, size=size, stride=stride, pad=0)
    del T, labels                   # Free up memory
    # labels_to_ground_truth(label_patches[0], showfig=debug)
    # labels_to_ground_truth(label_patches[-1], showfig=debug)

    dataset = tf.data.Dataset.from_tensor_slices((patches, label_patches))
    del patches, label_patches      # Free up memory
    return dataset


if __name__ == "__main__":
    dataset = get_dataset_for_segmentation()
