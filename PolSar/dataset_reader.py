import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path
import scipy.io
import tikzplotlib
import tensorflow as tf
from cvnn.utils import standarize, randomize
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

cao_dataset_parameters = {
    'validation_split': 0.1,  # Section 3.3.2
    'batch_size': 30,               # Section 3.3.2
    'sliding_window_size': 128,     # Section 3.3.2
    'sliding_window_stride': 25     # Section 3.3.2
}

OBER_COLORS = np.array([
    [1, 0.349, 0.392],  # Red; Built-up Area
    [0.086, 0.858, 0.576],  # Green; Wood Land
    [0.937, 0.917, 0.352],  # Yellow; Open Area
    [0, 0.486, 0.745]
])

# https://imagecolorpicker.com/en
FLEVOLAND = np.array([
    [255, 0, 0],        # Red; Steambeans
    [90, 11, 226],      # Purple; Peas
    [0, 131, 74],       # Green; Forest
    [0, 252, 255],      # Teal; Lucerne
    [255, 182, 228],    # Pink; Wheat
    [184, 0, 255],      # Magenta; Beet
    [254, 254, 0],      # Yellow; Potatoes
    [170, 138, 79],     # Brown; Bare Soil
    [1, 254, 3],        # Light green; Grass
    [255, 127, 0],      # Orange; Rapeseed
    [146, 0, 1],        # Bordeaux; Barley
    [191, 191, 255],    # Lila; Wheat 2
    [191, 255, 192],    # Marine Green; Wheat 3
    [0, 0, 254],        # Blue; Water
    [255, 217, 160]     # Beige; Buildings
])
FLEVOLAND = np.divide(FLEVOLAND, 255.0).astype(np.float32)

FLEVOLAND_2 = np.array([
    [255, 128, 0],      # Orange; Potatoes
    [138, 42, 116],      # Dark Purple; Fruit
    [0, 0, 255],        # Blue; Oats
    [255, 0, 0],        # Red; Beet
    [120, 178, 215],    # Light Blue; Barley
    [0, 102, 255],      # Middle Blue; Onions
    [251, 232, 45],     # Yellow; Wheat
    [1, 255, 3],        # Light green; Beans
    [204, 102, 225],    # Magenta; Peas
    [0, 204, 102],      # Green; Maize
    [204, 255, 204],    # Palid Green; Flax
    [204, 1, 102],      # Bordeaux; Rapeseed
    [255, 204, 204],    # Beige; Gress
    [102, 0, 204],      # Purple; Bare Soil

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


def to_real(data, labels):
    stacked = tf.stack([tf.math.real(data), tf.math.imag(data)], axis=-1)
    reshaped = tf.reshape(stacked, shape=tf.concat([tf.shape(data)[:-1], tf.convert_to_tensor([-1])], axis=-1))
    return reshaped, labels


def get_dataset_with_labels_t6(path: str, labels: str, debug=False):
    T, labels = open_dataset_t6(path, labels)
    if debug:
        labels_to_ground_truth(labels, showfig=True)
    labels = sparse_to_categorical_2D(labels)
    return T, labels


def get_dataset_with_labels_t3(dataset_path: str, labels: str):
    """
    Returns the t3 data with it's labels. It also checks matrices sizes agrees.
    :param dataset_path: The path with the .bin t3 files of the form T11_imag.bin, T11_real.bin, etc.
        with also .hdr files
    :param labels: A np 2D matrix of the labels in sparse mode. Where 0 is unlabeled
    :return: t3 matrix and labels in categorical mode (3D with the 3rd dimension size of number of classes)
    """
    labels_flev = sparse_to_categorical_2D(labels)
    t3 = open_dataset_t3(dataset_path)
    assert check_dataset_and_lebels(t3, labels_flev)
    return t3, labels_flev


def get_dataset_for_cao_segmentation(T, labels, complex_mode=True):
    train_dataset, test_dataset = get_dataset_for_segmentation(T=T, labels=labels,
                                                               size=cao_dataset_parameters['sliding_window_size'],
                                                               stride=cao_dataset_parameters['sliding_window_stride'],
                                                               test_size=cao_dataset_parameters['validation_split'])
    train_dataset = train_dataset.batch(cao_dataset_parameters['batch_size']).map(flip)
    test_dataset = test_dataset.batch(cao_dataset_parameters['batch_size'])
    if not complex_mode:
        train_dataset = train_dataset.map(to_real)
        test_dataset = test_dataset.map(to_real)
    return train_dataset, test_dataset


def get_dataset_for_segmentation(T, labels, size: int = 128, stride: int = 25, test_size: float = 0.2) -> \
        (tf.data.Dataset, tf.data.Dataset):
    """

    :param T:
    :param labels:
    :param size:
    :param stride:
    :param test_size:
    :return:
    """
    patches, label_patches = sliding_window_operation(T, labels, size=size, stride=stride, pad=0)
    del T, labels  # Free up memory
    # labels_to_ground_truth(label_patches[0], showfig=debug)
    # labels_to_ground_truth(label_patches[-1], showfig=debug)
    x_train, x_test, y_train, y_test = train_test_split(patches, label_patches,
                                                        test_size=test_size,
                                                        shuffle=True)

    del patches, label_patches  # Free up memory
    # dataset = tf.data.Dataset.from_tensor_slices((patches, label_patches)).shuffle(buffer_size=2500,
    #                                                                                reshuffle_each_iteration=False)
    # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    del x_train, y_train, x_test, y_test
    # set_trace()
    # check_no_coincidence(train_dataset, test_dataset)
    return train_dataset, test_dataset


def get_dataset_for_classification(path: str, labels: str):
    """
    Gets dataset ready to be processed for the mlp model.
    The dataset will be 2D dimensioned where the first element will be each pixel that will have 21 complex values each.
    :return: Tuple (T, labels)
    """
    T, labels = open_dataset_t6(path, labels)
    labels_to_ground_truth(labels)
    T, labels = remove_unlabeled(T, labels)
    labels -= 1  # map [1, 3] to [0, 2]
    T = T.reshape(-1, T.shape[-1])  # Image to 1D
    labels = labels.reshape(np.prod(labels.shape))
    return T, labels


def open_dataset_t3(path: str):
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


def check_dataset_and_lebels(dataset, labels):
    return dataset.shape[:2] == labels.shape[:2]


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


def labels_to_ground_truth(labels, showfig=False, savefig: Optional[str] = None, colors=None) -> np.ndarray:
    """
    Transforms the labels to a RGB format so it can be drawn as images
    :param labels: The labels to be transformed to rgb
    :param showfig: boolean. If true it will show the generated ground truth image
    :param savefig: boolean. If true it will save the generated ground truth image
    :param colors: Color palette to be used. Must be at least size of the labels. TODO: Some kind of check for this?
    :return: numpy array of the ground truth RGB image
    """
    if colors is None:
        if np.max(labels) == 3 or np.max(labels) == 4:
            print("Using Oberpfaffenhofen dataset colors")
            colors = OBER_COLORS
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
                ground_truth[i, j] = colors[labels[i, j] - 1]
    plt.imshow(ground_truth)
    if showfig:
        plt.show()
    if savefig is not None:
        assert isinstance(savefig, str)
        plt.imsave(savefig + ".pdf", ground_truth)
        tikzplotlib.save(savefig + ".tex")
    return ground_truth


def sparse_to_categorical_2D(labels) -> np.ndarray:
    classes = np.max(labels)
    ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                ground_truth[i, j, labels[i, j] - 1] = 1.
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
        lab = np.pad(lab, ((pad, pad), (pad, pad)))
    for x in range(0, im.shape[0] - size, stride):
        for y in range(0, im.shape[1] - size, stride):
            slice_x = slice(x, x + size)
            slice_y = slice(y, y + size)
            tiles.append(im[slice_x, slice_y])
            label_tiles.append(lab[slice_x, slice_y])
    assert np.all([p.shape == (size, size, im.shape[2]) for p in tiles])
    assert np.all([p.shape == (size, size, lab.shape[2]) for p in label_tiles])
    if not pad:  # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
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


if __name__ == "__main__":
    dataset = get_dataset_for_segmentation()
