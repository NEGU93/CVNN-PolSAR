import random
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path
import scipy.io
from pdb import set_trace
import tikzplotlib
import tensorflow as tf
from typing import Tuple, Optional, List, Union
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

zhang_dataset_parameters = {
    'validation_split': 0.1,
    'test_split': 0.9,
    'batch_size': 100,
    'sliding_window_size': 12,
    'sliding_window_stride': 1
}

OBER_COLORS = np.array([
    [1, 0.349, 0.392],  # Red; Built-up Area
    [0.086, 0.858, 0.576],  # Green; Wood Land
    [0.937, 0.917, 0.352],  # Yellow; Open Area
    [0, 0.486, 0.745]
])

BRET_COLORS = np.array([
    [0.086, 0.858, 0.576],  # Green; Forest
    [0, 0.486, 0.745],  # Blue; Piste
    [1, 0.349, 0.392],  # Red; Built-up Area
    [0.937, 0.917, 0.352],  # Yellow; Open Area
])

SF_COLORS = {
    "SF-ALOS2": [
        [132, 112, 255],
        [0, 0, 255],
        [0, 255, 0],
        [192, 0, 0],
        [0, 255, 255],
        [255, 255, 0]
    ],
    "SF-GF3": [
        [132, 112, 255],
        [0, 0, 255],
        [0, 255, 0],
        [192, 0, 0],
        [0, 255, 255],
        [255, 255, 0]
    ],
    "SF-RISAT": [
        [132, 112, 255],
        [0, 0, 255],
        [0, 255, 0],
        [192, 0, 0],
        [0, 255, 255],
        [255, 255, 0]
    ],
    "SF-RS2": [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [255, 255, 0],
        [255, 0, 255]
    ],
    "SF-AIRSAR": [
        [0, 255, 255],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0]
    ]
}

COLORS = {"BRETIGNY": BRET_COLORS, "OBER": OBER_COLORS, **SF_COLORS}

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
    # data = tf.image.random_flip_up_down(data)

    return data, labels


def sparse_to_categorical_1D(labels) -> np.ndarray:
    classes = np.max(labels)
    ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
    for i in range(labels.shape[0]):
        if labels[i] != 0:
            ground_truth[i, labels[i] - 1] = 1.
    return ground_truth


def pauli_rgb_map_plot(labels, dataset_name: str, t: Optional[np.ndarray] = None, path=None, mask=None):
    labels_rgb = labels_to_rgb(labels, colors=COLORS[dataset_name], mask=mask)
    fig, ax = plt.subplots()
    if t is not None:
        rgb = np.stack([t[:, :, 0], t[:, :, 1], t[:, :, 2]], axis=-1).astype(np.float32)
        ax.imshow(rgb)
    ax.imshow(labels_rgb, alpha=0.4)
    if path is not None:
        path = str(path)
        if len(path.split(".")) < 2:
            path = path + ".png"
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)


def labels_to_rgb(labels, showfig=False, savefig: Optional[str] = None, colors=None, mask=None, format: str = '.png') \
        -> np.ndarray:
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
        labels = np.argmax(labels, axis=-1) + 1
    elif len(labels.shape) != 2:
        raise ValueError(f"Expected labels to be rank 3 or 2, received rank {len(labels.shape)}.")
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
    ground_truth = np.zeros(labels.shape + (3,), dtype=float if np.max(colors) <= 1 else np.uint8)  # 3 channels for RGB
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


class PolsarDatasetHandler(ABC):

    def __init__(self, name: str, mode: str, complex_mode: bool = True, real_mode: str = 'real_imag',
                 normalize: bool = False, balance_dataset: bool = False):
        self.name = name
        assert mode.lower() in {"s", "t"}
        self.mode = mode.lower()
        self.real_mode = real_mode.lower()
        self.complex_mode = complex_mode
        self.image, self.labels, self.sparse_labels = self.open_image()
        assert self.image.shape[:2] == self.labels.shape[:2]
        if normalize:
            self.image, _ = tf.linalg.normalize(self.image, axis=[0, 1])
        if balance_dataset:
            self._balance_image()
        self.weights = self._get_weights(self.labels)

    @abstractmethod
    def open_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_dataset(self, method: str, percentage: Union[Tuple[float], float] = 0.2,
                    size: int = 128, stride: int = 25, shuffle: bool = True, pad=0,
                    savefig: Optional[str] = None, orientation: str = "vertical", data_augment: bool = False,
                    batch_size: int = cao_dataset_parameters['batch_size'], task: str = "segmentation"):
        percentage = self._parse_percentage(percentage)
        if method == "random":
            x_patches, y_patches = self._get_shuffled_dataset(size=size, stride=stride, pad=pad, percentage=percentage,
                                                              shuffle=shuffle)
        elif method == "separate":
            x_patches, y_patches = self._get_separated_dataset(percentage=percentage, size=size, stride=stride, pad=pad,
                                                               savefig=savefig, orientation=orientation,
                                                               shuffle=shuffle)
        elif method == "single_separated_image":
            x_patches, y_patches = self._get_single_image_separated_dataset(percentage=percentage, savefig=savefig,
                                                                            orientation=orientation, pad=True)
        else:
            raise ValueError(f"Unknown dataset method {method}")
        assert task.lower() in {"classification", "segmentation"}
        if task.lower() == "classification":
            assert method != "single_separated_image", f"Can't apply classification to the full image."
            self.labels = np.reshape(self.labels[:, size // 2, size // 2, :],
                                     shape=(self.labels.shape[0], self.labels.shape[-1]))
        ds_list = [self._transform_to_tensor(x, y, batch_size=batch_size,
                                             data_augment=data_augment if i == 0 else False)
                   for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        return ds_list

    """
        PRIVATE
    """

    @staticmethod
    def _parse_percentage(percentage) -> Tuple[float]:
        if isinstance(percentage, int):
            assert percentage == 1
            percentage = (1.,)
        if isinstance(percentage, float):
            if percentage == 1:
                percentage = (1.,)
            elif 0 < percentage < 1:
                percentage = (1 - percentage, percentage)
            else:
                raise ValueError(f"Percentage must be 0 < percentage < 1, received {percentage}")
        else:
            percentage = tuple(percentage)
            assert sum(percentage) == 1., f"percentage must add to 1, " \
                                          f"but it adds to sum{percentage} = {sum(percentage)}"
        return percentage

    @staticmethod
    def _pad_image(image, labels):
        first_dim_pad = int(2 ** 5 * np.ceil(image.shape[0] / 2 ** 5)) - image.shape[0]
        second_dim_pad = int(2 ** 5 * np.ceil(image.shape[1] / 2 ** 5)) - image.shape[1]
        paddings = [
            [int(np.ceil(first_dim_pad / 2)), int(np.floor(first_dim_pad / 2))],
            [int(np.ceil(second_dim_pad / 2)), int(np.floor(second_dim_pad / 2))],
            [0, 0]
        ]
        image = tf.pad(image, paddings)
        labels = tf.pad(labels, paddings)
        return image, labels

    def _slice_dataset(self, percentage: tuple, orientation: str, savefig: Optional[str]):
        orientation = orientation.lower()
        percentage = self._parse_percentage(percentage)

        if orientation == "horizontal":
            total_length = self.image.shape[1]
        elif orientation == "vertical":
            total_length = self.image.shape[0]
        else:
            raise ValueError(f"Orientation {orientation} unknown.")

        th = 0
        x_slice = []
        y_slice = []
        mask_slice = []
        for per in percentage:
            slice_1 = slice(th, th + int(total_length * per))
            th += int(total_length * per)
            if orientation == "horizontal":
                x_slice.append(self.image[:, slice_1])
                y_slice.append(self.labels[:, slice_1])
                mask_slice.append(self.sparse_labels[:, slice_1])
            else:
                x_slice.append(self.image[slice_1])
                y_slice.append(self.labels[slice_1])
                mask_slice.append(self.sparse_labels[slice_1])
        if savefig:
            slices_names = [
                'train_ground_truth', 'val_ground_truth', 'test_ground_truth'
            ]
            for i, y in enumerate(y_slice):
                self.print_ground_truth(label=y, t=x_slice[i], mask=mask_slice[i], path=str(savefig) + slices_names[i])
        return x_slice, y_slice

    def _transform_to_tensor(self, x, y, batch_size: int, data_augment: bool = False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.batch(batch_size)
        if data_augment:
            ds = ds.map(flip)
        if not self.complex_mode:
            ds = ds.map(lambda img, labels: transform_to_real_map_function(img, labels, self.real_mode))
        return ds

    # Get dataset
    def _get_shuffled_dataset(self, size: int = 128, stride: int = 25, percentage: Tuple[float] = (0.8, 0.2),
                              shuffle: bool = True, pad=0) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Applies the sliding window operations getting smaller images of a big image T.
        Splits dataset into train and test.
        :param size: Size of the window to be used on the sliding window operation.
        :param stride:
        :param percentage: float. Percentage of examples to be used for the test set [0, 1]
        :return: a Tuple of tf.Datasets (train_dataset, test_dataset)
        """
        patches, label_patches = self.sliding_window_operation(self.image, self.labels, size=size, stride=stride,
                                                               pad=pad)
        # del T, labels  # Free up memory
        x_train = patches
        y_train = label_patches
        x = []
        y = []
        for per in percentage[1:]:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=per, shuffle=shuffle)
            x.append(x_test)
            y.append(y_test)
        del patches, label_patches
        x_train, y_train = self._remove_empty_image(data=x_train, labels=y_train)
        x.insert(0, x_train)
        y.insert(0, y_train)
        return x, y

    def _get_separated_dataset(self, percentage: tuple, size: int = 128, stride: int = 25, shuffle: bool = True, pad=0,
                               savefig: Optional[str] = None, orientation: str = "vertical"):
        images, labels = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)

        # train_slice_label = _balance_image(train_slice_label)
        self.weights = self._get_weights(labels[0])
        for i in range(0, len(labels)):
            images[i], labels[i] = self.sliding_window_operation(images[i], labels[i],
                                                                 size=size, stride=stride, pad=pad)
            if i == 0:
                images[i], labels[i] = self._remove_empty_image(data=images[i], labels=labels[i])
                if shuffle:  # No need to shuffle the rest
                    images[i], labels[i] = sklearn.utils.shuffle(images[i], labels[i])
        return images, labels

    def _get_single_image_separated_dataset(self, percentage: tuple, savefig: Optional[str] = None,
                                            orientation: str = "vertical", pad: bool = False):
        x, y = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)

        self.weights = self._get_weights(y[0])
        for i in range(0, len(y)):
            if pad:
                x[i], y[i] = self._pad_image(x[i], y[i])
            x[i] = tf.expand_dims(x[i], axis=0)
            y[i] = tf.expand_dims(y[i], axis=0)
        return x, y

    @staticmethod
    def _remove_empty_image(data, labels):
        filtered_data = []
        filtered_labels = []
        for i in range(0, len(labels)):
            if not np.all(labels[i] == 0):
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
        filtered_data = np.array(filtered_data)
        filtered_labels = np.array(filtered_labels)
        return filtered_data, filtered_labels

    # BALANCE DATASET
    @staticmethod
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

    def _balance_image(self):
        """
        Removes pixel labels so that all classes have the same amount of classes (balance dataset)
        Balance dataset of one-hot encoded labels
        """
        classes = tf.argmax(self.labels, axis=-1)
        mask = np.all((self.labels == tf.zeros(shape=self.labels.shape[-1])), axis=-1)
        classes = tf.where(mask, classes, classes + 1)  # Increment classes, now 0 = no label
        totals = [tf.math.reduce_sum((classes == cls).numpy().astype(int)).numpy() for cls in
                  range(1, tf.math.reduce_max(classes).numpy() + 1)]
        if all(element == totals[0] for element in totals):
            print("All labels are already balanced")
            return self.labels
        min_value = min(totals)
        for value in range(1, len(totals) + 1):
            matrix_mask = self._select_random(classes, value=value, total=min_value)
            self.labels = tf.where(tf.expand_dims((classes != value).numpy() | matrix_mask, axis=-1),
                                   self.labels, tf.zeros(shape=self.labels.shape))

    @staticmethod
    def _get_weights(labels):
        classes = tf.argmax(labels, axis=-1)
        mask = np.all((labels == tf.zeros(shape=labels.shape[-1])), axis=-1)
        classes = tf.where(mask, classes, classes + 1)  # Increment classes, now 0 = no label
        totals = [tf.math.reduce_sum((classes == cls).numpy().astype(int)).numpy() for cls in
                  range(1, tf.math.reduce_max(classes).numpy() + 1)]
        return max(totals) / totals

    """
        PUBLIC
    """

    # Utils
    @staticmethod
    def sliding_window_operation(im, lab, size: int = 128, stride: int = 25, pad: int = 0, segmentation: bool = True) \
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
        if pad:
            im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)))
            lab = np.pad(lab, ((pad, pad), (pad, pad), (0, 0)))
        assert im.shape[0] > size and im.shape[1] > size
        for x in range(0, im.shape[0] - size + 1, stride):
            for y in range(0, im.shape[1] - size + 1, stride):
                slice_x = slice(x, x + size)
                slice_y = slice(y, y + size)
                tiles.append(im[slice_x, slice_y])
                if segmentation:
                    label_tiles.append(lab[slice_x, slice_y])
                else:
                    label_tiles.append(lab[x + int(size / 2), y + int(size / 2)])
        assert np.all([p.shape == (size, size, im.shape[2]) for p in tiles])
        # assert np.all([p.shape == (size, size, lab.shape[2]) for p in label_tiles])
        if not pad:  # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
            # import pdb; pdb.set_trace()
            assert int(np.shape(tiles)[0]) == int(
                (np.floor((im.shape[0] - size) / stride) + 1) * (np.floor((im.shape[1] - size) / stride) + 1))
        return np.array(tiles), np.array(label_tiles)

    @staticmethod
    def sparse_to_categorical_2D(labels) -> np.ndarray:
        classes = np.max(labels)
        ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != 0:
                    ground_truth[i, j, labels[i, j] - 1] = 1.
        return ground_truth

    # Open with path
    @staticmethod
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

    @staticmethod
    def open_s_dataset(path: str):
        path = Path(path)
        # http://www.spectralpython.net/fileio.html#envi-headers
        s_11_meta = envi.open(path / 's11.bin.hdr', path / 's11.bin')
        s_12_meta = envi.open(path / 's12.bin.hdr', path / 's12.bin')
        s_21_meta = envi.open(path / 's21.bin.hdr', path / 's21.bin')
        s_22_meta = envi.open(path / 's22.bin.hdr', path / 's22.bin')

        s_11 = s_11_meta.read_band(0)
        s_12 = s_12_meta.read_band(0)
        s_21 = s_21_meta.read_band(0)
        s_22 = s_22_meta.read_band(0)

        assert np.all(s_21 == s_12)

        return np.stack((s_11, s_12, s_22), axis=-1)

    @staticmethod
    def open_dataset_t6(path: str):
        path = Path(path)
        first_read = standarize(envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0))
        T = np.zeros(first_read.shape + (21,), dtype=complex)

        # Diagonal
        T[:, :, 0] = first_read
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
        return T

    # Debug
    def print_ground_truth(self, label: Optional = None, path=None, t=None, mask: Optional = None):
        if label is None:
            label = self.labels
        if mask is None:
            mask = self.sparse_labels
        return pauli_rgb_map_plot(label, mask=mask, dataset_name=self.name, t=t if self.mode == "t" else None,
                                  path=path)
