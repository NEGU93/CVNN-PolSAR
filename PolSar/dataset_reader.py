import os.path
import timeit
from scipy.ndimage import uniform_filter
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path
from pdb import set_trace
import tikzplotlib
import tensorflow as tf
from typing import Tuple, Optional, List, Union
from sklearn.model_selection import train_test_split
import sklearn
from cvnn.utils import transform_to_real_map_function, REAL_CAST_MODES

BUFFER_SIZE = 32000

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
        [0, 1., 1.],  # Light blue (bare soil)
        [1., 1., 0],  # Yellow (Mountain)
        [0, 0, 1.],  # Blue (Water)
        [1., 0, 0],  # Red (Buildings)
        [0, 1., 0]  # Green (Vegetation)
    ]
}

# https://imagecolorpicker.com/en
FLEVOLAND_15 = np.array([
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
FLEVOLAND_15 = np.divide(FLEVOLAND_15, 255.0).astype(np.float32)

FLEVOLAND_14 = np.array([
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
FLEVOLAND_14 = np.divide(FLEVOLAND_14, 255.0).astype(np.float32)

COLORS = {"BRET": BRET_COLORS, "OBER": OBER_COLORS,
          "FLEVOLAND": FLEVOLAND_15, "FLEVOLAND_14": FLEVOLAND_14,
          **SF_COLORS}

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

SUPPORTED_MODES = {"s", "t", "k"}


# Tensorflow Methods
def flip(data, labels):
    """
    Flip augmentation
    :param data: Image to flip
    :param labels: Image labels
    :return: Augmented image
    """
    data = tf.image.random_flip_left_right(data)
    # data = tf.image.random_flip_up_down(data)     # TODO: Select the contrary axis

    return data, labels


def mean_filter(input, filter_size=3):
    """
    Performs mean filter on an image with a filter of size (filter_size, filter_size)
    :param input: Image of shape (Height, Width, Channels)
    :param filter_size:
    :return:
    """
    input_transposed = tf.transpose(input, perm=[2, 0, 1])  # Get channels to the start 9xhxw
    filter = tf.ones(shape=(filter_size, filter_size, 1, 1), dtype=input.dtype.real_dtype)
    filtered_T_real = tf.nn.convolution(input=tf.expand_dims(tf.math.real(input_transposed), axis=-1),
                                        filters=filter, padding="SAME")
    filtered_T_imag = tf.nn.convolution(input=tf.expand_dims(tf.math.imag(input_transposed), axis=-1),
                                        filters=filter, padding="SAME")
    filtered_T = tf.complex(filtered_T_real, filtered_T_imag)
    my_filter_result = tf.transpose(tf.squeeze(filtered_T), perm=[1, 2, 0])  # Get channels to the end again
    my_filter_result = my_filter_result / (filter_size * filter_size)
    # This method would be better but it has problems with tf 2.4.1 (used on conda-develop)
    # tf_mean_real = tfa.image.mean_filter2d(tf.math.real(input), filter_shape=filter_size, padding="CONSTANT")
    # tf_mean_imag = tfa.image.mean_filter2d(tf.math.imag(input), filter_shape=filter_size, padding="CONSTANT")
    # tf_mean = tf.complex(tf_mean_real, tf_mean_imag)
    # assert np.allclose(tf_mean, my_filter_result)
    if filter_size == 1:
        assert np.all(my_filter_result == input), "mean filter of size 1 changed the input matrix"
    return my_filter_result


def _remove_lower_part(coh):
    mask = np.array(
        [True, True, True,
         False, True, True,
         False, False, True]
    )
    masked_coh = tf.boolean_mask(coh, mask, axis=2)
    return masked_coh


def sparse_to_categorical_1D(labels) -> np.ndarray:
    classes = np.max(labels)
    ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
    for i in range(labels.shape[0]):
        if labels[i] != 0:
            ground_truth[i, labels[i] - 1] = 1.
    return ground_truth


def pauli_rgb_map_plot(labels, dataset_name: str, t: Optional[np.ndarray] = None, path=None, mask=None, ax=None):
    labels_rgb = labels_to_rgb(labels, colors=COLORS[dataset_name], mask=mask)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    alpha = 1.
    # set_trace()
    if t is not None:
        alpha = 0.8
        rgb = np.stack([t[:, :, 0], t[:, :, 1], t[:, :, 2]], axis=-1).astype(np.float32)
        ax.imshow(rgb)
    ax.imshow(labels_rgb, alpha=alpha)
    if fig is not None and path is not None:
        path = str(path)
        if len(path.split(".")) < 2:
            path = path + ".png"
        fig.savefig(path)
    else:
        plt.show()
    if fig is not None:
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
            colors = FLEVOLAND_15
        elif np.max(labels) == 14:
            print("Using Flevoland 2 dataset colors")
            colors = FLEVOLAND_14
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


def transform_to_real_with_numpy(image, label, mode: str = "real_imag"):
    if mode not in REAL_CAST_MODES:
        raise KeyError(f"Unknown real cast mode {mode}")
    if mode == 'real_imag':
        ret_value = np.concatenate([np.real(image), np.imag(image)], axis=-1)
    elif mode == 'amplitude_phase':
        ret_value = np.concatenate([np.abs(image), np.angle(image)], axis=-1)
    elif mode == 'amplitude_only':
        ret_value = np.abs(image)
    elif mode == 'real_only':
        ret_value = np.real(image)
    else:
        raise KeyError(f"Real cast mode {mode} not implemented")
    return ret_value, label


class PolsarDatasetHandler(ABC):

    def __init__(self, root_path: str, name: str, mode: str, complex_mode: bool = True, real_mode: str = 'real_imag',
                 balance_dataset: bool = False, coh_kernel_size: int = 1):
        """

        :param root_path:
        :param name:
        :param mode:
        :param complex_mode:
        :param real_mode:
        :param balance_dataset: If classification, it will have a balanced dataset classes on the training set
            - For Bretigny it will load the balanced labels also so that even for segmentation it is balanced.
        :param coh_kernel_size:
        """
        self.root_path = Path(str(root_path))
        self.name = name
        self.coh_kernel_size = coh_kernel_size
        assert mode.lower() in {"s", "t", "k"}
        self.mode = mode.lower()
        self.real_mode = real_mode.lower()
        self.complex_mode = complex_mode  # TODO: Best practice to leave it outside
        self.balance_dataset = balance_dataset
        self._image = None
        self._sparse_labels = None
        self._labels = None
        self._labels_occurrences = None

    @property
    def image(self):
        if self._image is None:
            self._image = self.get_image()
        return self._image

    @property
    def sparse_labels(self):
        if self._sparse_labels is None:
            self._sparse_labels = self.get_sparse_labels()
        return self._sparse_labels

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.sparse_to_categorical_2D(self.sparse_labels)
        return self._labels

    @property
    def labels_occurrences(self):
        if self._labels_occurrences is None:
            self._labels_occurrences = self._get_occurrences(self.labels)
        return self._labels_occurrences

    """
        METHODS TO BE IMPLEMENTED
    """

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Must open the image. It must be:
            - numpy array
            - Data type np.complex
            - Shape (Width, Height, channels), with channels = 3 if self.mode = 'k' or 's'
                and channels = 6 if self.mode = 't'

            S format: (s_11, s_12, s_22) or equivalently (HH, HV, VV)
            T format:
        :return: The opened numpy image.
        """
        pass

    @abstractmethod
    def get_sparse_labels(self) -> np.ndarray:
        """
        Must open the labels in sparse mode (last dimension is a number from 0 to num_classes-1).
        :return: Numpy array with the sparse labels
        """
        pass

    """
        PUBLIC API
    """

    def get_dataset(self, method: str, percentage: Union[Tuple[float], float] = 0.2,
                    size: int = 128, stride: int = 25, shuffle: bool = True, pad="same",
                    savefig: Optional[str] = None, orientation: str = "vertical", data_augment: bool = False,
                    remove_last: bool = False, classification: bool = False,
                    batch_size: int = cao_dataset_parameters['batch_size'], use_tf_dataset=False):
        """
        Get the dataset in the desired form
        :param method: One of
            - 'random': Sample patch images randomly using sliding window operation (swo).
            - 'separate': Splits the image according to `percentage` parameter. Then gets patches using swo.
            - 'single_separated_image': Splits the image according to `percentage` parameter. Returns full image.
        :param percentage: Tuple giving the dataset split percentage.
            If sum(percentage) != 1 it will add an extra value to force sum(percentage) = 1.
            If sum(percentage) > 1 or it has at least one negative value it will raise an exception.
            Example, for 60% train, 20% validation and 20% test set, use percentage = (.6, .2, .2) or (.6, .2).
        :param size: Size of generated patches images. By default it will generate images of 128x128.
        :param stride: Stride used for the swo. If stride < size, parches will have coincident pixels.
        :param shuffle: Shuffle image patches (ignored if method == 'single_separated_image')
        :param pad: Pad image before swo or just add padding to output for method == 'single_separated_image'
        :param savefig: Used only if method='single_separated_image'.
            - It shaves len(percentage) images with the cropped generated images.
        :param orientation: Cut the image 'horizontally' or 'vertically' when split (using percentage param for sizes).
            Ignored if method == 'random'
        :param data_augment: Only used if use_tf_dataset = True. It performs data aumentation using flip.
        :param remove_last:
        :param classification: If true, it will have only one value per image path.
            Example, for a train dataset of shape (None, 128, 128, 3):
                classification = True: labels will be of shape (None, classes)
                classification = False: labels will be of shape (None, 128, 128, classes)
        :param batch_size:
        :param use_tf_dataset: If True, return dtype will be a tf.Tensor dataset instead of numpy array.
        :return: Returns a list of [train, (validation), (test), (k-folds)] according to percentage parameter.
            - Each list[i] is a tuple of (data, labels) where both data and labels are numpy arrays.
        """
        if method == "random":
            x_patches, y_patches = self._get_shuffled_dataset(size=size, stride=stride, pad=pad, percentage=percentage,
                                                              shuffle=shuffle, remove_last=remove_last,
                                                              classification=classification)
        elif method == "separate":
            x_patches, y_patches = self._get_separated_dataset(percentage=percentage, size=size, stride=stride, pad=pad,
                                                               savefig=savefig, orientation=orientation,
                                                               shuffle=shuffle,
                                                               classification=classification)
        elif method == "single_separated_image":
            assert not classification, f"Can't apply classification to the full image."
            x_patches, y_patches = self._get_single_image_separated_dataset(percentage=percentage, savefig=savefig,
                                                                            orientation=orientation, pad=True)
        else:
            raise ValueError(f"Unknown dataset method {method}")
        if use_tf_dataset:
            ds_list = [self._transform_to_tensor(x, y, batch_size=batch_size,
                                                 data_augment=data_augment if i == 0 else False, shuffle=shuffle)
                       for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        else:
            if self.complex_mode:
                ds_list = [(x, y) for i, (x, y) in enumerate(zip(x_patches, y_patches))]
            else:
                ds_list = [transform_to_real_with_numpy(x, y, self.real_mode)
                           for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        return tuple(ds_list)

    def print_ground_truth(self, label: Optional = None, path=None, t=None,
                           mask: Optional[Union[bool, np.ndarray]] = None, ax=None):
        """
        Saves or shows the labels rgb map.
        :param label: Labels to be printed as RGB map. If None it will use the dataset labels.
        :param path: Path where to save the image
        :param t:  TODO: Fix this
        :param mask: (Optional) One of
            - Boolean array with the same shape as label. False values will be printed as black.
            - If True: It will use self label to remove non labeled pixels from images
        :param ax: (Optional) axis where to plot the new image, used for overlapping figures.
        :return: None
        """
        if label is None:
            label = self.labels
        if isinstance(mask, bool) and mask:
            mask = self.get_sparse_labels()
        return pauli_rgb_map_plot(label, mask=mask, dataset_name=self.name, t=t if self.mode == "t" else None,
                                  path=path, ax=ax)

    def print_image_png(self, savefile: bool = False, showfig: bool = False, img_name: str = "PauliRGB.png"):
        coh_matrix = self.get_coherency_matrix(kernel_shape=1)
        rgb_image = self._coh_to_rgb(coh_matrix=coh_matrix)
        if showfig:
            plt.imshow(rgb_image)
            plt.show()
        if savefile:
            plt.imsave(self.root_path / img_name, np.clip(rgb_image, a_min=0., a_max=1.))

    """
        GETTERS
    """

    def get_pauli_vector(self):
        if self.mode == 'k':
            return self.image
        elif self.mode == 's':
            return self._get_k_vector(HH=self.image[:, :, 0], VV=self.image[:, :, 1], HV=self.image[:, :, 2])
        elif self.mode == 't':
            raise Exception("It is not possible to obtain the pauli vector from the coherency matrix")
        else:
            raise ValueError(f"Mode {self.mode} not supported. Supported modes: {SUPPORTED_MODES}")

    def get_coherency_matrix(self, kernel_shape=3):
        if self.mode == 't':
            return self.image
        elif self.mode == 'k':
            return self._numpy_coh_from_k(self.image, kernel_shape=kernel_shape)
        elif self.mode == 's':
            return self.numpy_coh_matrix(HH=self.image[:, :, 0], VV=self.image[:, :, 2], HV=self.image[:, :, 1],
                                         kernel_shape=kernel_shape)
        else:
            raise ValueError(f"Mode {self.mode} not supported. Supported modes: {SUPPORTED_MODES}")

    """
        PRIVATE
    """

    # 3 get dataset main methods
    def _get_shuffled_dataset(self, size: int = 128, stride: int = 25,
                              percentage: Union[Tuple[float], float] = (0.8, 0.2),
                              shuffle: bool = True, pad="same", remove_last: bool = False,
                              classification: bool = False) -> (np.ndarray, np.ndarray):
        """
        Applies the sliding window operations getting smaller images of a big image T.
        Splits dataset into train and test.
        :param size: Size of the window to be used on the sliding window operation.
        :param stride:
        :param percentage: float. Percentage of examples to be used for the test set [0, 1]
        :return: a Tuple of np.array (train_dataset, test_dataset)
        """
        patches, label_patches = self.apply_sliding_on_self_data(size=size, stride=stride, pad=pad,
                                                                 classification=classification)
        x, y = self._separate_dataset(patches=patches, label_patches=label_patches, classification=classification,
                                      percentage=percentage, shuffle=shuffle, remove_last=remove_last)
        return x, y

    def _get_separated_dataset(self, percentage: tuple, size: int = 128, stride: int = 25, shuffle: bool = True, pad=0,
                               savefig: Optional[str] = None, orientation: str = "vertical", classification=False):
        images, labels = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)
        # train_slice_label = _balance_image(train_slice_label)
        for i in range(0, len(labels)):
            images[i], labels[i] = self.apply_sliding(images[i], labels[i], size=size, stride=stride, pad=pad,
                                                      classification=classification)
            if shuffle:  # No need to shuffle the rest as val and test does not really matter they are shuffled
                images[i], labels[i] = sklearn.utils.shuffle(images[i], labels[i])
        return images, labels

    def _get_single_image_separated_dataset(self, percentage: tuple, savefig: Optional[str] = None,
                                            orientation: str = "vertical", pad: bool = False):
        x, y = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)
        for i in range(0, len(y)):
            if pad:
                x[i], y[i] = self._pad_image(x[i], y[i])
            x[i] = np.expand_dims(x[i], axis=0)
            y[i] = np.expand_dims(y[i], axis=0)
        return x, y

    # Parser/check input
    @staticmethod
    def _parse_percentage(percentage) -> List[float]:
        if isinstance(percentage, int):
            assert percentage == 1
            percentage = (1.,)
        if isinstance(percentage, float):
            if percentage == 1.:
                percentage = (percentage,)
            if 0 < percentage < 1:
                percentage = (percentage, 1 - percentage)
            else:
                raise ValueError(f"Percentage must be 0 < percentage < 1, received {percentage}")
        else:
            percentage = list(percentage)
            assert all(p >= 0. for p in percentage), "percentage elements can't be negative"
            assert sum(percentage) <= 1., f"percentage must add to 1 max, " \
                                          f"but it adds to sum({percentage}) = {sum(percentage)}"
            if sum(percentage) < 1:
                percentage = percentage + [1 - sum(percentage)]
        return percentage

    @staticmethod
    def _parse_pad(pad, kernel_size):
        if isinstance(pad, int):
            pad = ((pad, pad), (pad, pad))
        elif isinstance(pad, str):
            if pad.lower() == "same":
                pad = tuple([(w // 2, (w - 1) // 2) for w in kernel_size])
            elif pad.lower() == "valid":
                pad = ((0, 0), (0, 0))
            else:
                raise ValueError(f"padding: {pad} not recognized. Possible values are 'valid' or 'same'")
        else:
            pad = list(pad)
            assert len(pad) == 2
            for indx in range(2):
                if isinstance(pad[indx], int):
                    pad[indx] = (pad[indx], pad[indx])
                else:
                    pad[indx] = tuple(pad[indx])
                    assert len(pad[indx]) == 2
            pad = tuple(pad)
        return pad

    # Methods to print rgb image
    def _coh_to_rgb(self, coh_matrix):
        diagonal = coh_matrix[:, :, :3].astype(np.float32)
        diagonal[diagonal == 0] = float("nan")
        diag_db = 10 * np.log10(diagonal)
        noramlized_diag = np.zeros(shape=diag_db.shape)
        noramlized_diag[:, :, 0] = self.normalize_without_outliers(diag_db[:, :, 0])
        noramlized_diag[:, :, 2] = self.normalize_without_outliers(diag_db[:, :, 1])
        noramlized_diag[:, :, 1] = self.normalize_without_outliers(diag_db[:, :, 2])
        return noramlized_diag

    @staticmethod
    def remove_outliers(data, iqr=(1, 99)):
        low = np.nanpercentile(data, iqr[0])
        high = np.nanpercentile(data, iqr[1])
        return low, high

    def normalize_without_outliers(self, data):
        low, high = self.remove_outliers(data)
        return (data - low) / (high - low)

    # Methods with Tensorflow
    @staticmethod
    def _pad_image(image, labels):
        first_dim_pad = int(2 ** 5 * np.ceil(image.shape[0] / 2 ** 5)) - image.shape[0]
        second_dim_pad = int(2 ** 5 * np.ceil(image.shape[1] / 2 ** 5)) - image.shape[1]
        paddings = [
            [int(np.ceil(first_dim_pad / 2)), int(np.floor(first_dim_pad / 2))],
            [int(np.ceil(second_dim_pad / 2)), int(np.floor(second_dim_pad / 2))],
            [0, 0]
        ]
        image = tf.pad(image, paddings)  # TODO: Do it with other than tf?
        labels = tf.pad(labels, paddings)
        return image, labels

    def _transform_to_tensor(self, x, y, batch_size: int, data_augment: bool = False, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=BUFFER_SIZE)
        ds = ds.batch(batch_size)
        if data_augment:
            ds = ds.map(flip)
        if not self.complex_mode:
            ds = ds.map(lambda img, labels: transform_to_real_map_function(img, labels, self.real_mode))
        return ds

    # MISC
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
                self.print_ground_truth(label=y, t=x_slice[i], mask=mask_slice[i],
                                        path=str(savefig) + slices_names[i])
        return x_slice, y_slice

    @staticmethod
    def balanced_test_split(x_all, y_all, test_size, shuffle):
        x_train_per_class, x_test_per_class, y_train_per_class, y_test_per_class = [], [], [], []
        sparse_y = np.argmax(y_all, axis=-1)
        for cls in range(y_all.shape[-1]):
            x_train, x_test, y_train, y_test = train_test_split(x_all[sparse_y == cls], y_all[sparse_y == cls],
                                                                train_size=int(
                                                                    (1 - test_size) * y_all.shape[0] / y_all.shape[-1]),
                                                                shuffle=shuffle)
            x_train_per_class.append(x_train)
            x_test_per_class.append(x_test)
            y_train_per_class.append(y_train)
            y_test_per_class.append(y_test)
        x_train = np.concatenate(x_train_per_class)
        x_test = np.concatenate(x_test_per_class)
        y_train = np.concatenate(y_train_per_class)
        y_test = np.concatenate(y_test_per_class)
        return x_train, x_test, y_train, y_test

    def _separate_dataset(self, patches, label_patches, percentage: Union[Tuple[float], float], shuffle: bool = True,
                          classification: bool = False,
                          remove_last: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Separates dataset patches according to the percentage.
        :param percentage: list of percentages for each value,
            example [0.9, 0.02, 0.08] to get 90% train, 2% val and 8% test.
        :param shuffle: Shuffle dataset before split.
        :return: tuple of two lists of size = len(percentage), one with data x and other with labels y.
        """
        percentage = self._parse_percentage(percentage)
        x_test = patches
        y_test = label_patches
        x = []
        y = []
        for i, per in enumerate(percentage[:-1]):
            if classification and self.balance_dataset:
                x_train, x_test, y_train, y_test = self.balanced_test_split(x_test, y_test, test_size=1 - per,
                                                                            shuffle=shuffle)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=1 - per,
                                                                    shuffle=True if classification else shuffle,
                                                                    stratify=y_test if classification else None)
            percentage[i + 1:] = [value / (1 - percentage[i]) for value in percentage[i + 1:]]
            x.append(x_train)
            y.append(y_train)
        if not remove_last:
            x.append(x_test)
            y.append(y_test)
        return x, y

    @staticmethod
    def _to_classification(x, y, remove_unlabeled=False):
        y = np.reshape(y[:, y.shape[1] // 2, y.shape[2] // 2, :], newshape=(y.shape[0], y.shape[-1]))
        # assert [np.all(y_patches_class[i][:] == y_patches[i][0][0][:]) for i in range(len(y_patches_class))]
        # 2. Remove empty pixels
        if remove_unlabeled:  # TODO: Remove this?
            mask = np.invert(np.all(y == 0, axis=-1))
            x = x[mask]
            y = y[mask]
        return x, y

    @staticmethod
    def _remove_empty_image(data, labels):
        if len(labels.shape) == 4:
            mask = np.invert(np.all(np.all(labels == 0, axis=-1), axis=(1, 2)))
        elif len(labels.shape) == 2:
            mask = np.invert(np.all(labels == 0, axis=-1))
        else:
            raise ValueError(f"Ups, shape of labels of size {len(labels.shape)} not supported.")
        masked_filtered_data = np.reshape(data[mask], newshape=(-1,) + data.shape[1:])
        masked_filtered_labels = labels[mask]
        return masked_filtered_data, masked_filtered_labels

    @staticmethod
    def _sliding_window_operation(im, lab, size: Tuple[int, int], stride: int,
                                  pad: Tuple[Tuple[int, int], Tuple[int, int]],
                                  segmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
        im = np.pad(im, (pad[0], pad[1], (0, 0)))
        lab = np.pad(lab, (pad[0], pad[1], (0, 0)))
        assert im.shape[0] > size[0] and im.shape[1] > size[1], f"Image shape ({im.shape[0]}x{im.shape[1]}) " \
                                                                f"is smaller than the window to apply " \
                                                                f"({size[0]}x{size[1]})"
        for x in range(0, im.shape[0] - size[0] + 1, stride):
            for y in range(0, im.shape[1] - size[1] + 1, stride):
                slice_x = slice(x, x + size[0])
                slice_y = slice(y, y + size[1])
                tiles.append(im[slice_x, slice_y])
                if segmentation:
                    label_tiles.append(lab[slice_x, slice_y])
                else:
                    label_tiles.append(lab[x + int(size[0] / 2), y + int(size[1] / 2)])
        assert np.all([p.shape == (size[0], size[1], im.shape[2]) for p in tiles])
        # assert np.all([p.shape == (size, size, lab.shape[2]) for p in label_tiles])
        # if not pad:  # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
        #     assert int(np.shape(tiles)[0]) == int(
        #         (np.floor((im.shape[0] - size[0]) / stride) + 1) * (np.floor((im.shape[1] - size[1]) / stride) + 1))
        # print(f"tiles shape before going out of sliding window op {np.array(tiles).shape}")
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
        first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)

        # Diagonal
        T[:, :, 0] = first_read
        T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
        T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)

        # Upper part
        T[:, :, 3] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
        T[:, :, 4] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
        T[:, :, 5] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
        return T.astype(np.complex64)

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

        return np.stack((s_11, s_12, s_22), axis=-1).astype(np.complex64)

    @staticmethod
    def open_dataset_t6(path: str):
        path = Path(path)
        first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (21,), dtype=np.complex64)

        # Diagonal
        T[:, :, 0] = first_read
        T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
        T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)
        T[:, :, 3] = envi.open(path / 'T44.bin.hdr', path / 'T44.bin').read_band(0)
        T[:, :, 4] = envi.open(path / 'T55.bin.hdr', path / 'T55.bin').read_band(0)
        T[:, :, 5] = envi.open(path / 'T66.bin.hdr', path / 'T66.bin').read_band(0)

        T[:, :, 6] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
        T[:, :, 7] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
        T[:, :, 8] = envi.open(path / 'T14_real.bin.hdr', path / 'T14_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T14_imag.bin.hdr', path / 'T14_imag.bin').read_band(0)
        T[:, :, 9] = envi.open(path / 'T15_real.bin.hdr', path / 'T15_real.bin').read_band(0) + \
                     1j * envi.open(path / 'T15_imag.bin.hdr', path / 'T15_imag.bin').read_band(0)
        T[:, :, 10] = envi.open(path / 'T16_real.bin.hdr', path / 'T16_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T16_imag.bin.hdr', path / 'T16_imag.bin').read_band(0)

        T[:, :, 11] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
        T[:, :, 12] = envi.open(path / 'T24_real.bin.hdr', path / 'T24_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T24_imag.bin.hdr', path / 'T24_imag.bin').read_band(0)
        T[:, :, 13] = envi.open(path / 'T25_real.bin.hdr', path / 'T25_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T25_imag.bin.hdr', path / 'T25_imag.bin').read_band(0)
        T[:, :, 14] = envi.open(path / 'T26_real.bin.hdr', path / 'T26_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T26_imag.bin.hdr', path / 'T26_imag.bin').read_band(0)

        T[:, :, 15] = envi.open(path / 'T34_real.bin.hdr', path / 'T34_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T34_imag.bin.hdr', path / 'T34_imag.bin').read_band(0)
        T[:, :, 16] = envi.open(path / 'T35_real.bin.hdr', path / 'T35_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T35_imag.bin.hdr', path / 'T35_imag.bin').read_band(0)
        T[:, :, 17] = envi.open(path / 'T36_real.bin.hdr', path / 'T36_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T36_imag.bin.hdr', path / 'T36_imag.bin').read_band(0)

        T[:, :, 18] = envi.open(path / 'T45_real.bin.hdr', path / 'T45_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T45_imag.bin.hdr', path / 'T45_imag.bin').read_band(0)
        T[:, :, 19] = envi.open(path / 'T46_real.bin.hdr', path / 'T46_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T46_imag.bin.hdr', path / 'T46_imag.bin').read_band(0)

        T[:, :, 20] = envi.open(path / 'T56_real.bin.hdr', path / 'T56_real.bin').read_band(0) + \
                      1j * envi.open(path / 'T56_imag.bin.hdr', path / 'T56_imag.bin').read_band(0)
        return T.astype(np.complex64)

    """
        Input format conversion
    """

    @staticmethod
    def _get_k_vector(HH, VV, HV):
        k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
        # tf.transpose(k, perm=[1, 2, 0], conjugate=False)
        return np.transpose(k, axes=[1, 2, 0])

    def _get_coherency_matrix(self, HH, VV, HV, kernel_shape=3):
        # Section 2: https://earth.esa.int/documents/653194/656796/LN_Advanced_Concepts.pdf
        print("WARNING: _get_coherency_matrix is deprected. Use numpy_coh_matrix instead.")
        k = self._get_k_vector(HH=HH, VV=VV, HV=HV)
        tf_k = tf.expand_dims(k, axis=-1)  # From shape hxwx3 to hxwx3x1
        T_mat = tf.linalg.matmul(tf_k, tf_k,
                                 adjoint_b=True)  # k * k^H: inner 2 dimensions specify valid matrix multiplication dim
        one_channel_T = tf.reshape(T_mat, shape=(
            T_mat.shape[0], T_mat.shape[1], T_mat.shape[2] * T_mat.shape[3]))  # hxwx3x3 to hxwx9
        removed_lower_part_T = _remove_lower_part(one_channel_T)  # hxwx9 to hxwx6 removing lower part of matrix
        filtered_T = mean_filter(removed_lower_part_T, kernel_shape)
        reorder = tf.transpose(np.array([
            filtered_T[:, :, 0], filtered_T[:, :, 3], filtered_T[:, :, 5],
            filtered_T[:, :, 1], filtered_T[:, :, 2], filtered_T[:, :, 4]
        ]), perm=[1, 2, 0])
        return reorder.numpy()

    def numpy_coh_matrix(self, HH, VV, HV, kernel_shape=3):
        k = self._get_k_vector(HH=HH, VV=VV, HV=HV)
        return self._numpy_coh_from_k(k, kernel_shape=kernel_shape)

    @staticmethod
    def _numpy_coh_from_k(k, kernel_shape=3):
        np_k = np.expand_dims(k, axis=-1)
        t_mat = np.matmul(np_k, np.transpose(np.conjugate(np_k), axes=[0, 1, 3, 2]))
        one_channel_T = np.reshape(t_mat, newshape=(t_mat.shape[0], t_mat.shape[1], t_mat.shape[2] * t_mat.shape[3]))
        mask = np.array(
            [True, True, True,
             False, True, True,
             False, False, True]
        )
        removed_lower_part_T = one_channel_T[:, :, mask]
        filtered_T = removed_lower_part_T
        ordered_filtered_t = np.transpose(
            np.array([
                filtered_T[:, :, 0], filtered_T[:, :, 3], filtered_T[:, :, 5],
                filtered_T[:, :, 1], filtered_T[:, :, 2], filtered_T[:, :, 4]
            ]), axes=[1, 2, 0])
        return uniform_filter(ordered_filtered_t, mode="constant",  # TODO: use constant mode?
                              size=(kernel_shape,) * (len(ordered_filtered_t.shape) - 1) + (1,))

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
        flatten_img = np.reshape(img, np.prod(img.shape))
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

    @staticmethod
    def _get_occurrences(labels):
        classes = tf.argmax(labels, axis=-1)
        mask = np.all((labels == tf.zeros(shape=labels.shape[-1])), axis=-1)
        classes = tf.where(mask, classes, classes + 1)  # Increment classes, now 0 = no label
        totals = [tf.math.reduce_sum((classes == cls).numpy().astype(int)).numpy() for cls in
                  range(1, tf.math.reduce_max(classes).numpy() + 1)]
        return max(totals) / totals

    """
        PUBLIC
    """

    def apply_sliding_on_self_data(self, *args, **kwargs):
        return self.apply_sliding(image=self.image, labels=self.labels, *args, **kwargs)

    def apply_sliding(self, image, labels, size: Union[int, Tuple[int, int]] = 128, stride: int = 25, pad="same",
                      classification: bool = False, remove_unlabeled: bool = True):
        """
        Performs the sliding window operation to the image.
        :param size:
        :param stride:
        :param pad:
        :param classification:
        :param remove_unlabeled:
        :param use_saved_image:
        :param save_generated_images:
        :return: image and label patches of the main image and labels
        """
        # TODO: Removing save images as it colides with different methods.
        # if not hasattr(self, "patches"):
        # Parse input params
        use_saved_image = False
        save_generated_images = False
        if isinstance(size, int):
            size = (size, size)
        else:
            size = tuple(size)
            assert len(size) == 2
        pad = self._parse_pad(pad, size)
        # Get folder and file name
        temp_path = self.root_path / "dataset_preprocess_cache"
        os.makedirs(str(temp_path), exist_ok=True)
        config_string = f"{'cls' if classification else 'seg'}_{self.name.lower()}_{self.mode}" \
                        f"_window{size}_stride{stride}_pad{pad}"
        # Do I already performed the math operations?
        if use_saved_image and os.path.isfile(temp_path / (config_string + "_patches.npy")):
            print(f"Loading dataset {config_string}_patches.npy")
            start = timeit.default_timer()
            patches = np.load(str(temp_path / (config_string + "_patches.npy"))).astype(np.complex64)
            label_patches = np.load(str(temp_path / (config_string + "_labels.npy")))
            print(f"Load done in {timeit.default_timer() - start} seconds")
        else:
            print(f"Computing dataset {config_string}_patches.npy")
            start = timeit.default_timer()
            patches, label_patches = self._sliding_window_operation(image, labels, size=size, stride=stride, pad=pad)
            if remove_unlabeled:
                patches, label_patches = self._remove_empty_image(data=patches, labels=label_patches)
            # print(f"patches shape after sliding window op{patches.shape}")
            if save_generated_images and not \
                    os.path.exists(str(temp_path / ("seg" + config_string[3:] + "_patches.npy"))):
                np.save(str(temp_path / ("seg" + config_string[3:] + "_patches.npy")), patches.astype(np.complex64))
                np.save(str(temp_path / ("seg" + config_string[3:] + "_labels.npy")), label_patches)
            if classification:
                patches, label_patches = self._to_classification(x=patches, y=label_patches,
                                                                 remove_unlabeled=remove_unlabeled)
                if save_generated_images:
                    np.save(str(temp_path / ("cls" + config_string[3:] + "_patches.npy")), patches.astype(np.complex64))
                    np.save(str(temp_path / ("cls" + config_string[3:] + "_labels.npy")), label_patches)
            print(f"Computation done in {timeit.default_timer() - start} seconds")
        return patches, label_patches

