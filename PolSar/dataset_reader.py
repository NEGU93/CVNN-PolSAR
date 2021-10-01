import random

import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path
import scipy.io
from pdb import set_trace
import tikzplotlib
import tensorflow as tf
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import sklearn
from cvnn.utils import standarize, randomize, transform_to_real_map_function

cao_dataset_parameters = {
    'validation_split': 0.1,  # Section 3.3.2
    'batch_size': 30,  # Section 3.3.2
    'sliding_window_size': 128,  # Section 3.3.2
    'sliding_window_stride': 25,  # Section 3.3.2
    'window_for_mlp': 32  # Section 3.4
}

OBER_COLORS = np.array([
    [1, 0.349, 0.392],      # Red; Built-up Area
    [0.086, 0.858, 0.576],  # Green; Wood Land
    [0.937, 0.917, 0.352],  # Yellow; Open Area
    [0, 0.486, 0.745]
])

BRET_COLORS = np.array([
    [0.086, 0.858, 0.576],  # Green; Forest
    [0, 0.486, 0.745],      # Blue; Piste
    [1, 0.349, 0.392],      # Red; Built-up Area
    [0.937, 0.917, 0.352],  # Yellow; Open Area
])

# https://imagecolorpicker.com/en
FLEVOLAND = np.array([
    [255, 0, 0],  # Red; Steambeans
    [90, 11, 226],  # Purple; Peas
    [0, 131, 74],  # Green; Forest
    [0, 252, 255],  # Teal; Lucerne
    [255, 182, 228],  # Pink; Wheat
    [184, 0, 255],  # Magenta; Beet
    [254, 254, 0],  # Yellow; Potatoes
    [170, 138, 79],  # Brown; Bare Soil
    [1, 254, 3],  # Light green; Grass
    [255, 127, 0],  # Orange; Rapeseed
    [146, 0, 1],  # Bordeaux; Barley
    [191, 191, 255],  # Lila; Wheat 2
    [191, 255, 192],  # Marine Green; Wheat 3
    [0, 0, 254],  # Blue; Water
    [255, 217, 160]  # Beige; Buildings
])
FLEVOLAND = np.divide(FLEVOLAND, 255.0).astype(np.float32)

FLEVOLAND_2 = np.array([
    [255, 128, 0],  # Orange; Potatoes
    [138, 42, 116],  # Dark Purple; Fruit
    [0, 0, 255],  # Blue; Oats
    [255, 0, 0],  # Red; Beet
    [120, 178, 215],  # Light Blue; Barley
    [0, 102, 255],  # Middle Blue; Onions
    [251, 232, 45],  # Yellow; Wheat
    [1, 255, 3],  # Light green; Beans
    [204, 102, 225],  # Magenta; Peas
    [0, 204, 102],  # Green; Maize
    [204, 255, 204],  # Palid Green; Flax
    [204, 1, 102],  # Bordeaux; Rapeseed
    [255, 204, 204],  # Beige; Gress
    [102, 0, 204],  # Purple; Bare Soil

])
FLEVOLAND_2 = np.divide(FLEVOLAND_2, 255.0).astype(np.float32)

DEFAULT_PLOTLY_COLORS = [
    [31, 119, 180],  # Blue
    [255, 127, 14],  # Orange
    [44, 160, 44],  # Green
    [214, 39, 40],
    [148, 103, 189], [140, 86, 75],
    [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207]
]
DEFAULT_PLOTLY_COLORS = np.divide(DEFAULT_PLOTLY_COLORS, 255.0).astype(np.float32)


def flip(data, labels):
    """
    Flip augmentation
    :param data: Image to flip
    :param labels: Image labels
    :return: Augmented image
    """
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)

    return data, labels


def check_dataset_and_lebels(dataset, labels):
    return dataset.shape[:2] == labels.shape[:2]


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


def remove_unlabeled(x, y):
    """
    Removes the unlabeled pixels from both image and labels
    :param x: image input
    :param y: labels inputs. all values of 0's will be eliminated and its corresponging values of x
    :return: tuple (x, y) without the unlabeled pixels.
    """
    mask = y != 0
    return x[mask], y[mask]


def sparse_to_categorical_2D(labels) -> np.ndarray:
    classes = np.max(labels)
    ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                ground_truth[i, j, labels[i, j] - 1] = 1.
    return ground_truth


def sparse_to_categorical_1D(labels) -> np.ndarray:
    classes = np.max(labels)
    ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
    for i in range(labels.shape[0]):
        if labels[i] != 0:
            ground_truth[i, labels[i] - 1] = 1.
    return ground_truth


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


def sliding_window_operation(im, lab, size: int = 128, stride: int = 25, pad: int = 0) \
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
        lab = np.pad(lab, ((pad, pad), (pad, pad), (0, 0)))
    for x in range(0, im.shape[0] - size + 1, stride):
        for y in range(0, im.shape[1] - size + 1, stride):
            slice_x = slice(x, x + size)
            slice_y = slice(y, y + size)
            tiles.append(im[slice_x, slice_y])
            label_tiles.append(lab[slice_x, slice_y])
    assert np.all([p.shape == (size, size, im.shape[2]) for p in tiles])
    assert np.all([p.shape == (size, size, lab.shape[2]) for p in label_tiles])
    if not pad:  # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
        # import pdb; pdb.set_trace()
        assert int(np.shape(tiles)[0]) == int(
            (np.floor((im.shape[0] - size) / stride) + 1) * (np.floor((im.shape[1] - size) / stride) + 1))
    return np.array(tiles), np.array(label_tiles)


def load_image_train(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def check_no_coincidence(train_dataset, test_dataset):
    for data, label in test_dataset:
        for train_data, train_label in train_dataset:
            assert not np.array_equal(data.numpy(), train_data.numpy())


def remove_unlabeled_and_flatten(T, labels, shift_map=True):
    T, labels = remove_unlabeled(T, labels)
    if shift_map:
        labels -= 1  # map [1, 3] to [0, 2]
    T = T.reshape(-1, T.shape[-1])  # Image to 1D
    labels = labels.reshape(np.prod(labels.shape))
    return T, labels


def remove_unlabeled_with_window(T, labels, window_size=32):
    """
    Returns a flatten dataset of only labeled pixels with surrounded pixels according to window_size
    :param T:
    :param labels: ATTENTION: Must be sparse!
    :param window_size:
    :return:
    """
    results = []
    results_labels = []
    pad = int(window_size / 2)
    T = np.pad(T, ((pad, pad), (pad, pad), (0, 0)))
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            if labels[i, j] != 0:
                results.append(T[i:i + window_size, i:i + window_size].flatten())
                results_labels.append(labels[i, j])
    return np.array(results), np.array(results_labels)


def _transform_to_tensor(x, y, data_augment: bool = True,
                         batch_size: int = cao_dataset_parameters['batch_size'], complex_mode: bool = True,
                         mode: str = 'real_imag'):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    if data_augment:
        ds = ds.map(flip)
    if not complex_mode:
        ds = ds.map(lambda img, labels: transform_to_real_map_function(img, labels, mode))
    return ds


def _parse_percentage(percentage):
    try:
        percentage = tuple(percentage)
    except Exception as e:
        raise ValueError(f"Could not cast {percentage} to tuple")
    assert np.all([percentage[i] > 0 for i in range(0, len(percentage))]), "No negative percentage values allowed."
    if len(percentage) == 3:
        assert sum(percentage) == 1., "Total percentage does not get to 100%"
    elif len(percentage) == 2:
        assert sum(percentage) < 1., "Total percentage over 100%"
    else:
        raise ValueError(f"Expected percentage length 2 or 3. Received {len(percentage)}")
    return percentage


def _select_random(img, value: int, total: int):
    """
    Gets an image with different values and returns a boolean matrix with a total number of True equal to the total
        parameter where all True are located in the same place where the img has the value passed as parameter
    :param img: Image to be selected
    :param value: Value to be matched
    :param total:
    :return:
    """
    assert len(img.shape) == 2
    flatten_img = np.reshape(img, tf.math.reduce_prod(img.shape).numpy())
    random_indx = np.random.permutation(len(flatten_img))
    mixed_img = flatten_img[random_indx]
    selection_mask = np.zeros(shape=img.shape)
    selected = 0
    indx = 0
    saved_indexes = []
    while selected < total:
        val = mixed_img[indx]
        if val == value:
            selected += 1
            saved_indexes.append(random_indx[indx])
            assert img[int(random_indx[indx] / img.shape[1])][random_indx[indx] % img.shape[1]] == value
            selection_mask[int(random_indx[indx] / img.shape[1])][random_indx[indx] % img.shape[1]] = 1
        indx += 1
    return selection_mask.astype(bool)


def _balance_image(labels):
    """
    Removes pixel labels so that all classes have the same amount of classes (balance dataset)
    :param labels: One-hot encoded labels (rank 3 is forced due to _select_random function)
    :return: Balance dataset of one-hot encoded labels
    """
    classes = tf.argmax(labels, axis=-1)
    mask = np.all((labels == tf.zeros(shape=labels.shape[-1])), axis=-1)
    classes = tf.where(mask, classes, classes+1)    # Increment classes, now 0 = no label
    totals = [tf.math.reduce_sum((classes == cls).numpy().astype(int)).numpy() for cls in range(1, tf.math.reduce_max(classes).numpy()+1)]
    if all(element == totals[0] for element in totals):
        print("All labels are already balanced")
        return labels
    min_value = min(totals)
    for value in range(1, len(totals)+1):
        matrix_mask = _select_random(classes, value=value, total=min_value)
        labels = tf.where(tf.expand_dims((classes != value).numpy() | matrix_mask, axis=-1),
                          labels, tf.zeros(shape=labels.shape))
    return labels


"""----------
-   Public  -
----------"""


def labels_to_ground_truth(labels, showfig=False, savefig: Optional[str] = None, colors=None, mask=None,
                           format: str = '.png') -> np.ndarray:
    """
    Transforms the labels to a RGB format so it can be drawn as images
    :param labels: The labels to be transformed to rgb
    :param showfig: boolean. If true it will show the generated ground truth image
    :param savefig: A string. String with the file to be saved of the generated ground truth image
    :param colors: Color palette to be used. Must be at least size of the labels. TODO: Some kind of check for this?
    :param mask: If the mask is passed it will remove the pixels (black) when mask == 0
    :return: numpy array of the ground truth RGB image
    """
    if len(labels.shape) == 3:
        # new_labels = np.zeros(labels.shape[:-1]).astype(int)
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         if np.any(labels[i][j] != 0):
        #             # try:
        #             new_labels[i][j] = int(np.nonzero(labels[i][j])[0] + 1)
        #             # except Exception as e:
        #             #     set_trace()
        # labels = new_labels
        labels = np.argmax(labels, axis=-1) + 1
    elif len(labels.shape) != 2:
        raise ValueError(f"Expected labels to be rank 3 or 2, received rank {len(labels.shape)}.")
    # import pdb; pdb.set_trace()
    if colors is None:
        if np.max(labels) == 3:
            print("Using Oberpfaffenhofen dataset colors")
            colors = OBER_COLORS
        elif np.max(labels) == 4:
            print("Using Bretigny dataset colors")
            colors = BRET_COLORS
        elif np.max(labels) == 15:
            print("Using Flevoland dataset colors")
            colors = FLEVOLAND
        elif np.max(labels) == 14:
            print("Using Flevoland 2 dataset colors")
            colors = FLEVOLAND_2
        else:
            print("Using Plotly dataset colors")
            colors = DEFAULT_PLOTLY_COLORS
    ground_truth = np.zeros(labels.shape + (3,), dtype=float)  # 3 channels for RGB
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                # try:
                ground_truth[i, j] = colors[labels[i, j] - 1]
                # except Exception as e:
                #     import pdb; pdb.set_trace()
    if mask is not None:
        ground_truth[mask == 0] = [0, 0, 0]
    if showfig:
        plt.imshow(ground_truth)
        # plt.show()
    if savefig is not None:
        assert isinstance(savefig, str)
        if format == ".tex":
            tikzplotlib.save(savefig + ".tex")
        else:
            plt.imsave(savefig + format, ground_truth)
    return ground_truth


def open_dataset_t3(path: str, labels: str):
    labels = scipy.io.loadmat(labels)['label']
    return open_t_dataset_t3(path), labels


# To open dataset .bin/hdr using the path
def open_dataset_t6(path: str, labels: str):
    labels = scipy.io.loadmat(labels)['label']
    path = Path(path)
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


def open_t_dataset_t3(path: str):
    path = Path(path)
    first_read = standarize(envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0))
    T = np.zeros(first_read.shape + (6,), dtype=complex)

    # Diagonal
    T[:, :, 0] = first_read
    T[:, :, 1] = standarize(envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0))
    T[:, :, 2] = standarize(envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0))

    # Upper part
    T[:, :, 3] = standarize(envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0))
    T[:, :, 4] = standarize(envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0))
    T[:, :, 5] = standarize(envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                            1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0))
    return T


def get_dataset_with_labels_t6(path: str, labels: str, debug=False):
    T, labels = open_dataset_t6(path, labels)
    if debug:
        labels_to_ground_truth(labels, showfig=True)
    labels = sparse_to_categorical_2D(labels)
    return T, labels


# Opens dataset .bin/hdr using the path and change sparse labels to categorical
def get_dataset_with_labels_t3(dataset_path: str, labels: str):
    """
    Returns the t3 data with it's labels. It also checks matrices sizes agrees.
    :param dataset_path: The path with the .bin t3 files of the form T11_imag.bin, T11_real.bin, etc.
        with also .hdr files
    :param labels: A np 2D matrix of the labels in sparse mode. Where 0 is unlabeled
    :return: t3 matrix and labels in categorical mode (3D with the 3rd dimension size of number of classes)
    """
    t3, labels = open_dataset_t3(dataset_path, labels)
    labels_flev = sparse_to_categorical_2D(labels)
    assert check_dataset_and_lebels(t3, labels_flev)
    return t3, labels_flev


# Returns a dataset T with labels in the correct form
def get_dataset_for_segmentation(T, labels, size: int = 128, stride: int = 25, test_size: float = 0.2,
                                 shuffle: bool = True, pad=0) -> (tf.data.Dataset, tf.data.Dataset):
    """
    Applies the sliding window operations getting smaller images of a big image T.
    Splits dataset into train and test.
    :param T: Big image T (3D)
    :param labels: labels for T
    :param size: Size of the window to be used on the sliding window operation.
    :param stride:
    :param test_size: float. Percentage of examples to be used for the test set [0, 1]
    :return: a Tuple of tf.Datasets (train_dataset, test_dataset)
    """
    labels = _balance_image(labels)
    patches, label_patches = sliding_window_operation(T, labels, size=size, stride=stride, pad=pad)
    # del T, labels  # Free up memory
    x_train, x_test, y_train, y_test = get_tf_dataset_split(patches, label_patches, test_size=test_size, shuffle=shuffle)
    del patches, label_patches
    return x_train, x_test, y_train, y_test


def get_dataset_for_classification(T, labels, shift_map=True, test_size=0.8):
    """
    Gets dataset ready to be processed for the mlp model.
    The dataset will be 2D dimensioned where the first element will be each pixel that will have 21 complex values each.
    :return: Tuple (train_dataset, test_dataset, val_dataset).
    """
    T, labels = remove_unlabeled_and_flatten(T, labels, shift_map)
    labels = sparse_to_categorical_1D(labels)
    return get_tf_dataset_split(T, labels, test_size=test_size)


def get_tf_dataset_split(T, labels, test_size=0.1, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(T, labels, test_size=test_size, shuffle=shuffle)
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # del x_train, y_train, x_test, y_test
    return x_train, x_test, y_train, y_test


"""--------------
-   HIGH API    -
--------------"""


def get_separated_dataset(T, labels, percentage: tuple, size: int = 128, stride: int = 25, shuffle: bool = True, pad=0,
                          savefig: Optional[str] = None, complex_mode: bool = True, mode: str = 'real_imag'):
    percentage = _parse_percentage(percentage)

    slice_1 = int(T.shape[1] * percentage[0])
    slice_2 = int(T.shape[1] * percentage[1]) + slice_1

    # labels_to_ground_truth(labels, savefig='./full_image')

    train_slice_t = T[:, :slice_1]
    train_slice_label = labels[:, :slice_1]
    val_slice_t = T[:, slice_1:slice_2]
    val_slice_label = labels[:, slice_1:slice_2]
    test_slice_t = T[:, slice_2:]
    test_slice_label = labels[:, slice_2:]
    if savefig:
        labels_to_ground_truth(train_slice_label, savefig=savefig + 'train_ground_truth')
        labels_to_ground_truth(val_slice_label, savefig=savefig + 'val_ground_truth')
        labels_to_ground_truth(test_slice_label, savefig=savefig + 'test_ground_truth')
    train_slice_label = _balance_image(train_slice_label)
    # train_slice_label = _balance_image(train_slice_label)
    # set_trace()

    train_patches, train_label_patches = sliding_window_operation(train_slice_t, train_slice_label,
                                                                  size=size, stride=stride, pad=pad)
    val_patches, val_label_patches = sliding_window_operation(val_slice_t, val_slice_label,
                                                              size=size, stride=stride, pad=pad)
    test_patches, test_label_patches = sliding_window_operation(test_slice_t, test_slice_label,
                                                                size=size, stride=stride, pad=pad)

    if shuffle:     # No need to shuffle the rest
        train_patches, train_label_patches = sklearn.utils.shuffle(train_patches, train_label_patches)

    ds_train = _transform_to_tensor(train_patches, train_label_patches, data_augment=True, batch_size=30,
                                    complex_mode=complex_mode, mode=mode)
    ds_val = _transform_to_tensor(val_patches, val_label_patches, data_augment=False, batch_size=30,
                                  complex_mode=complex_mode, mode=mode)
    ds_test = _transform_to_tensor(test_patches, test_label_patches, data_augment=False, batch_size=30,
                                   complex_mode=complex_mode, mode=mode)
    del train_slice_t, train_slice_label, val_slice_t, val_slice_label, test_slice_t, test_slice_label
    del train_patches, train_label_patches, val_patches, val_label_patches, test_patches, test_label_patches
    return ds_train, ds_val, ds_test


"""----------
-   CAO     -
----------"""


def get_dataset_for_cao_segmentation(T, labels, complex_mode=True, shuffle=True, pad=0, mode: str = 'real_imag'):
    x_train, x_test, y_train, y_test = get_dataset_for_segmentation(T=T, labels=labels,
                                                                    size=cao_dataset_parameters['sliding_window_size'],
                                                                    stride=cao_dataset_parameters[
                                                                        'sliding_window_stride'],
                                                                    test_size=cao_dataset_parameters[
                                                                        'validation_split'],
                                                                    shuffle=shuffle, pad=pad)
    train_dataset = _transform_to_tensor(x_train, y_train, data_augment=True, mode=mode,
                                         batch_size=cao_dataset_parameters['batch_size'], complex_mode=complex_mode)
    test_dataset = _transform_to_tensor(x_test, y_test, data_augment=False, mode=mode,
                                        batch_size=cao_dataset_parameters['batch_size'], complex_mode=complex_mode)
    del x_train, x_test, y_train, y_test
    return train_dataset, test_dataset


def get_dataset_for_cao_classification(T, labels, complex_mode=True):
    T, labels = remove_unlabeled_with_window(T, labels, window_size=cao_dataset_parameters['window_for_mlp'])
    labels = sparse_to_categorical_1D(labels)
    """
    train_dataset, test_dataset = get_tf_dataset_split(T=T, labels=labels,
                                                       test_size=cao_dataset_parameters['validation_split'])
    train_dataset = train_dataset.batch(cao_dataset_parameters['batch_size'])
    test_dataset = test_dataset.batch(cao_dataset_parameters['batch_size'])
    if not complex_mode:
        train_dataset = train_dataset.map(to_real)
        test_dataset = test_dataset.map(to_real)
    return train_dataset, test_dataset
    """
    x_train, x_test, y_train, y_test = train_test_split(T, labels, test_size=cao_dataset_parameters['validation_split'],
                                                        shuffle=True)
    return x_train, x_test, y_train, y_test
