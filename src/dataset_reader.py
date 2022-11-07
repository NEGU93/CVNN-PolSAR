import os.path
import random
import timeit
import sys
import logging
import pandas as pd
import pickle
from packaging import version
from collections import defaultdict
from scipy.ndimage import uniform_filter
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from random import sample
from pathlib import Path
from pdb import set_trace
from itertools import compress
import tikzplotlib
from bisect import insort, bisect
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Optional, List, Union, Sequence, Generator
from sklearn.model_selection import train_test_split
import sklearn
from cvnn.utils import transform_to_real_map_function, REAL_CAST_MODES

BUFFER_SIZE = 32000
PAD_TYPING = Union[int, str, Sequence]

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

GARON_COLORS = np.array([
    [0.937, 0.917, 0.352],  # Yellow; Open Area
    [0.086, 0.858, 0.576],  # Green; Forest
    [1, 0.349, 0.392],  # Red; Built-up Area
    [0, 0.486, 0.745],  # Blue; Piste
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
          "GARON": GARON_COLORS,
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


def labels_stats(label_patches):
    """
    Verifies:
        - Total pixels per class balanced too
    Raises assertion error if image is not balanced
    :param label_patches:
    :return:
    """
    # count = np.bincount(np.where(label_patches == 1)[-1])  # Count of total pixels
    # assert np.all(np.logical_or(count == count[np.nonzero(count)][0], count == 0))
    counter = defaultdict(lambda: {"total": 0, "mixed": 0, "solo images": 0, "mixed images": 0})
    for i, la in enumerate(label_patches):
        present_classes = np.where(la == 1)[-1]     # Find all classes (again, there will be at least one).
        assert len(present_classes)                 # No empty image are supposed to be here.
        all_equal = np.all(present_classes == present_classes[0])  # Are all classes the same one?
        if all_equal:                               # If only one class present, then add it to the counter
            counter[present_classes[0]]["total"] += len(present_classes)
            counter[present_classes[0]]["solo images"] += 1
        else:               # If mixed case, then it must be balanced itself
            for cls in set(present_classes):
                counter[cls]["total"] += np.sum(present_classes == cls)
                counter[cls]["mixed"] += np.sum(present_classes == cls)
                counter[cls]["mixed images"] += 1
    min_case = np.min([counter[i]["total"] for i in range(label_patches.shape[-1]) if counter[i]["total"] != 0])
    set_trace()

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


def pauli_rgb_map_plot(labels, dataset_name: str, t: Optional[np.ndarray] = None, path=None, mask=None, ax=None,
                       showfig: bool = False, alpha=.8):
    colors = None
    if dataset_name in COLORS.keys():
        colors = COLORS[dataset_name]
    labels_rgb = labels_to_rgb(labels, colors=colors, mask=mask)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()  # figsize=labels_rgb.shape[:2])
    # set_trace()
    if t is not None:
        rgb = np.stack([t[:, :, 0], t[:, :, 1], t[:, :, 2]], axis=-1).astype(np.float32)
        ax.imshow(rgb)
    ax.imshow(labels_rgb, alpha=alpha)
    if fig is not None and path is not None:
        path = str(path)
        if len(path.split(".")) < 2:
            path = path + ".png"
        # ax.imsave(path)
        ax.axis('off')
        fig.savefig(path, bbox_inches='tight', pad_inches=0)  # , dpi=1)
    if showfig:
        plt.show()  # dpi=1)
    if fig is not None:
        plt.close(fig)
    return labels_rgb


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


class CounterList(Sequence):
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def ordered_insertion_by_key(sequence, value, key):
    if version.parse(f"{sys.version_info.major}.{sys.version_info.minor}") >= version.parse("3.10"):
        insort(sequence, value, key=lambda d: d[key])
    else:
        bisect_index = bisect(CounterList(sequence, key=lambda c: c[key]), value[key])
        sequence.insert(bisect_index, value)


def multi_dimensional_string_to_list(data: str) -> List:
    if data.startswith("["):
        assert data.endswith(']')
        result_list = []


class PolsarDatasetHandler(ABC):

    def __init__(self, root_path: str, name: str, mode: str, coh_kernel_size: int = 1):
        """

        :param root_path:
        :param name:
        :param mode:
        :param coh_kernel_size:
        """
        self.root_path = Path(str(root_path))
        self.name = name
        self.coh_kernel_size = coh_kernel_size
        assert mode.lower() in {"s", "t", "k"}
        self._mode = mode.lower()
        self._image = None
        self._sparse_labels = None
        self._labels = None
        self._labels_occurrences = None
        self.azimuth = None  # TODO: Child can define it wil "horizontal" or "vertical"

    @property
    def image(self):
        if self._image is None:
            self._image = self.get_image()
        return self._image

    @property
    def shape(self):
        return self.image.shape[:-1]

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
            self._labels_occurrences = self.get_occurrences(self.labels, normalized=True)
        return self._labels_occurrences

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self._mode != mode:
            self._mode = mode.lower()
            if self._image is not None:
                self._image = self.get_image()  # Reload image

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
        Must open the labels in sparse mode (last dimension is a number from 0 to num_classes+1).
        :return: Numpy array with the sparse labels
        """
        pass

    """
        PUBLIC API
    """

    def get_dataset(self, method: str, percentage: Union[Tuple[float, ...], float],
                    size: int = 128, stride: int = 25, shuffle: bool = True, pad="same",
                    savefig: Optional[str] = None,
                    azimuth: Optional[str] = None, data_augment: bool = False, classification: bool = False,
                    complex_mode: bool = True, real_mode: str = "real_imag",
                    balance_dataset: Union[bool, Tuple[bool]] = False, re_process_data=False,
                    batch_size: int = cao_dataset_parameters['batch_size'], cast_to_np=False):
        # 1. Parse input
        percentage = self._parse_percentage(percentage)
        if azimuth is None:
            azimuth = self.azimuth
            if azimuth is None and method != "random":
                raise ValueError(f"This instance does not have an azimuth defined. "
                                 f"Please select it using the azimuth parameter")
        # 2. Get filename
        kwargs = locals()       # Normally ordered.
        # In recent versions of python, dicts are ordered but I do it just in case
        object_variable_dict = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        object_variable_dict.update(kwargs)  # add the parameters
        object_variable_dict["mode"] = self.mode
        del object_variable_dict["self"]
        del object_variable_dict["root_path"]
        del object_variable_dict["batch_size"]
        del object_variable_dict["cast_to_np"]
        del object_variable_dict["savefig"]
        del object_variable_dict["re_process_data"]
        cache_path = self.root_path / "cache"
        cache_path.mkdir(exist_ok=True)
        # assert len(args) == 0       # TODO: *args not used. This can have issues! I need to add *args to **kwargs
        filename = ''       # f'mode_{self.mode}_'
        for k, v in sorted(object_variable_dict.items()):
            filename += f"{k}_{v}_".replace('.', '').replace(' ', '').replace('[', '').replace(']', '').replace(',', '').replace('(', '').replace(')', '')
        filename = filename[:-1]                            # Remove trailing '_' and add extension
        # 3. If data does not exist. Create it
        if re_process_data or not (cache_path / ("k0_" + filename)).is_dir():        # Dataset was not created
            if re_process_data:
                tf.print("Creating dataset ignoring saved files")
            else:
                tf.print("Dataset not found. Creating dataset")
            ds_list = self.generate_data(method=method, percentage=percentage, size=size, stride=stride,
                                         shuffle=shuffle, pad=pad, savefig=savefig, azimuth=azimuth,
                                         data_augment=data_augment, classification=classification,
                                         complex_mode=complex_mode, real_mode=real_mode,
                                         balance_dataset=balance_dataset, batch_size=batch_size, use_tf_dataset=False)
            for subset_index, subset in enumerate(ds_list):
                np.savez(str(cache_path / (f"k{subset_index}_" + filename + ".npz")),
                         images=subset[0], labels=subset[1])
                tf_dataset = tf.data.Dataset.from_tensor_slices((subset[0], subset[1]))
                tf.data.experimental.save(dataset=tf_dataset, path=str(cache_path / (f"k{subset_index}_" + filename)))
                with open(str(cache_path / (f"k{subset_index}_" + filename + ".pickle")), 'wb') as file:
                    pickle.dump(tf_dataset.element_spec, file)
            del ds_list
        else:
            tf.print("Dataset found. Loading...")
        # Get dataset
        tf_dataset = []
        for subset_index in range(len(percentage)):
            if cast_to_np:
                loaded = np.load(str(cache_path / (f"k{subset_index}_" + filename + ".npz")))
                tensor_data = (loaded["images"], loaded["labels"])
            else:
                element_spec = pickle.load(open(str(cache_path / (f"k{subset_index}_" + filename + ".pickle")), 'rb'))
                tensor_data = tf.data.experimental.load(str(cache_path / (f"k{subset_index}_" + filename)),
                                                        element_spec=element_spec)\
                    .shuffle(buffer_size=1000, reshuffle_each_iteration=True)\
                    .batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            tf_dataset.append(tensor_data)
        tf.print("Dataset loaded")
        return tf_dataset

    def generate_data(self, method: str, percentage,
                      size: int = 128, stride: int = 25, shuffle: bool = True, pad="same",
                      savefig: Optional[str] = None,
                      azimuth: Optional[str] = None, data_augment: bool = False, classification: bool = False,
                      complex_mode: bool = True, real_mode: str = "real_imag",
                      balance_dataset: Union[bool, Tuple[bool]] = False,
                      batch_size: int = cao_dataset_parameters['batch_size'], use_tf_dataset=False):
        """
        Get the dataset in the desired form
        :param method: One of
            - 'random': Sample patch images randomly using sliding window operation (swo)
            - 'separate': Splits the image according to `percentage` parameter. Then gets patches using swo
            - 'single_separated_image': Splits the image according to `percentage` parameter. Returns full image
        :param percentage: Tuple giving the dataset split percentage
            If sum(percentage) != 1 it will add an extra value to force sum(percentage) = 1.
            If sum(percentage) > 1 or it has at least one negative value it will raise an exception.
            Example, for 60% train, 20% validation and 20% test set, use percentage = (.6, .2, .2) or (.6, .2).
        :param size: Size of generated patches images. By default, it will generate images of 128x128
        :param stride: Stride used for the swo. If stride < size, parches will have coincident pixels
        :param shuffle: Shuffle image patches (ignored if method == 'single_separated_image')
        :param pad: Pad image before swo or just add padding to output for method == 'single_separated_image'
        :param savefig: Used only if method='single_separated_image'.
            - It saves len(percentage) images with the cropped generated images
        :param azimuth: Cut the image 'horizontally' or 'vertically' when split (using percentage param for sizes).
            Ignored if method == 'random'
        :param data_augment: Only used if use_tf_dataset = True. It performs data augmentation using flip
        :param classification: If true, it will have only one value per image path
            Example, for a train dataset of shape (None, 128, 128, 3):
                classification = True: labels will be of shape (None, classes)
                classification = False: labels will be of shape (None, 128, 128, classes)
        :param complex_mode: (default = True). Whether to return the data in complex dtype or float
        :param real_mode: If complex_mode = False, this param is used to specify the float format. One of:
            - real_imag: Stack real and imaginary part
            - amplitude_phase: stack amplitude and phase
            - amplitude_only: output only the amplitude
            - real_only: output only the real part
        :param balance_dataset: Works very differently according to classification param:
            - If classification == False: Balanced using images that have only one class in it:
                Example: If 100 and 200 images with only class 1 and 2 respectively,
                    100 of the 200 will be eliminated to have 100 and 100.
                    If 1000 images have both classes, they will not be touched
            - If classification == True: It balances the train set leaving the test set unbalanced
        :param batch_size: Used only if use_tf_dataset = True. Fixes the batch size of the tf.Dataset
        :param use_tf_dataset: If True, return dtype will be a tf.Tensor dataset instead of numpy array
        :return: Returns a list of [train, (validation), (test), (k-folds)] according to percentage parameter
            - Each list[i] is a tuple of (data, labels) where both data and labels are numpy arrays
        """
        if method == "random":
            x_patches, y_patches = self._get_shuffled_dataset(size=size, stride=stride, pad=pad, percentage=percentage,
                                                              shuffle=shuffle, classification=classification,
                                                              balance_dataset=balance_dataset)
        elif method == "separate":
            x_patches, y_patches = self._get_separated_dataset(percentage=percentage, size=size, stride=stride,
                                                               savefig=savefig, azimuth=azimuth,
                                                               shuffle=shuffle, balance_dataset=balance_dataset,
                                                               classification=classification)
        elif method == "single_separated_image":
            assert not classification, f"Can't apply classification to the full image."
            x_patches, y_patches = self._get_single_image_separated_dataset(percentage=percentage, savefig=savefig,
                                                                            azimuth=azimuth,
                                                                            balance_dataset=balance_dataset, pad=True)
        else:
            raise ValueError(f"Unknown dataset method {method}")
        if use_tf_dataset:
            ds_list = [self._transform_to_tensor(x, y, batch_size=batch_size, complex_mode=complex_mode,
                                                 data_augment=data_augment if i == 0 else False, shuffle=shuffle)
                       for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        else:
            if complex_mode:
                ds_list = [(np.array(x), np.array(y)) for i, (x, y) in enumerate(zip(x_patches, y_patches))]
            else:
                ds_list = [transform_to_real_with_numpy(x, y, real_mode)
                           for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        return tuple(ds_list)

    def print_ground_truth(self, label: Optional = None, path: Optional[str] = None,
                           transparent_image: Union[bool, float, np.ndarray] = False,
                           mask: Optional[Union[bool, np.ndarray]] = None, ax=None, showfig: bool = False):
        """
        Saves or shows the labels rgb map.
        :param label: Labels to be printed as RGB map. If None it will use the dataset labels.
        :param path: Path where to save the image
        :param transparent_image: One of:
            - If True it will also print the rgb image to superposed with the labels.
            - float: alpha value for the plotted image (if True it will use default)
        :param mask: (Optional) One of
            - Boolean array with the same shape as label. False values will be printed as black.
            - If True: It will use self label to remove non labeled pixels from images
            - Ignored if label is None
        :param ax: (Optional) axis where to plot the new image, used for overlapping figures.
        :param showfig: Show figure
        :return: np array of the rgb ground truth image
        """
        if label is None:
            label = self.labels
            mask = True  # In this case, just mask the missing labels.
        if isinstance(mask, bool) and mask:
            mask = self.sparse_labels  # TODO: I can force padding here.
        # TODO: I can force padding here.
        alpha = 0.7
        if isinstance(transparent_image, np.ndarray):
            t = transparent_image
        else:
            t = self.print_image_png(savefile=False, showfig=False) if transparent_image else None
            if isinstance(transparent_image, float):
                alpha = transparent_image
        return pauli_rgb_map_plot(label, mask=mask, dataset_name=self.name, t=t, path=path, ax=ax, showfig=showfig,
                                  alpha=alpha)

    def print_image_png(self, savefile: Union[bool, str] = False, showfig: bool = False,
                        img_name: str = "PauliRGB.png"):
        """
        Generates the RGB image
        :param savefile: Where to save the image or not.
            - Bool: If True it will save the image at self.root_path
            - str: path where to save the image
        :param showfig: Show image
        :param img_name: Name of the generated image
        :return: Rge rgb image as numpy
        """
        if self.mode == "t":
            coh_matrix = self.get_coherency_matrix(kernel_shape=1)
            rgb_image = self._diag_to_rgb(diagonal=coh_matrix[:, :, :3])
        else:
            k_vector = self.get_pauli_vector()
            k_module = k_vector * np.conj(k_vector)
            rgb_image = self._diag_to_rgb(diagonal=k_module)
        fig = plt.figure()  # figsize=rgb_image.shape[:2])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgb_image)
        if showfig:
            plt.show()  # dpi=1)
        if savefile:
            path = self.root_path
            if isinstance(savefile, str):
                path = Path(savefile)
            # plt.imsave(path / img_name, np.clip(rgb_image, a_min=0., a_max=1.))
            if len(img_name.split(".")) < 2:
                img_name = img_name + ".png"
            # ax.imsave(path)
            fig.savefig(path / img_name, bbox_inches='tight', pad_inches=0, dpi=1)
        plt.close(fig)
        return rgb_image

    def get_occurrences(self, labels: Optional = None, normalized=True):  # TODO: Make this with numpy
        """
        Get the occurrences of each label
        :param labels: (Optional) if None it will return the occurrences of self labels.
        :param normalized: Normalized the output, for example, [20, 10] will be transformed to [2, 1]
            - This is used to obtain the weights of a penalized loss.
        :return: a list label-wise occurrences
        """
        if labels is None:
            labels = self.labels
        classes = tf.argmax(labels, axis=-1)
        mask = np.all((labels == tf.zeros(shape=labels.shape[-1])), axis=-1)
        classes = tf.where(mask, classes, classes + 1)  # Increment classes, now 0 = no label
        totals = [tf.math.reduce_sum((classes == cls).numpy().astype(int)).numpy() for cls in
                  range(1, tf.math.reduce_max(classes).numpy() + 1)]
        if normalized:
            # TODO: Did I fucked it here? I think I fixed it now but check loss
            # totals = np.divide(totals, min(totals))
            totals = np.divide(totals, np.sum(totals))
        return totals

    """
        GETTERS
    """

    def get_real_image(self, mode: str = "real_imag"):
        """
        Returns the real_valued image
        :param mode: How to concat the real image output:
            - 'real_image': real and imaginary part
            - 'amplitude_phase': amplitude and phase
            - 'amplitude_only': only the amplitude
            - 'real_only': only the real part
        :return:
        """
        return transform_to_real_with_numpy(self.image, None, mode=mode)[0]

    @staticmethod
    def get_scattering_vector_from_k(pauli):
        scat = np.zeros(shape=pauli.shape, dtype=complex)  # k = HH + VV, HH - VV, HV
        scat[:, :, 0] = (pauli[:, :, 0] + pauli[:, :, 1]) / np.sqrt(2)  # s11, s12, s22 = HH, HV, VV
        scat[:, :, 2] = (pauli[:, :, 0] - pauli[:, :, 1]) / np.sqrt(2)
        scat[:, :, 1] = pauli[:, :, 2] / np.sqrt(2)
        return scat

    def get_scattering_vector(self):
        if self.mode == 's':
            return self.image
        elif self.mode == 'k':
            scat = self.get_scattering_vector_from_k(self.image)
            assert np.allclose(self._get_k_vector(HH=scat[:, :, 0], VV=scat[:, :, 2], HV=scat[:, :, 1]), self.image)
            return scat
        elif self.mode == 't':
            raise NotImplementedError("It is not possible to obtain the scattering vector from the coherency matrix")
        else:
            raise ValueError(f"Mode {self.mode} not supported. Supported modes: {SUPPORTED_MODES}")

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
                              shuffle: bool = True, pad: PAD_TYPING = "same", balance_dataset: bool = False,
                              classification: bool = False) -> (np.ndarray, np.ndarray):
        """
        Applies the sliding window operations getting smaller images of a big image T.
        Splits dataset into train and test.
        :param size: Size of the window to be used on the sliding window operation.
        :param stride:
        :param percentage: float. Percentage of examples to be used for the test set [0, 1]
        :return: a Tuple of np.array (train_dataset, test_dataset)
        """
        patches = self.apply_sliding_on_self_data(size=size, stride=stride, pad=pad, classification=classification)
        x, y = self._separate_dataset(patches=patches, classification=classification, percentage=percentage,
                                      shuffle=shuffle, balance_dataset=balance_dataset)
        # x = self.get_patches_image_from_point_and_self_image(patches_points=x, size=size, pad=pad)
        return x, y

    def _get_separated_dataset(self, percentage: tuple, size: int = 128, stride: int = 25, shuffle: bool = True,
                               savefig: Optional[str] = None, azimuth: Optional[str] = None, classification=False,
                               balance_dataset: Union[bool, Tuple[bool]] = False):
        images, labels = self._slice_dataset(percentage=percentage, savefig=savefig, azimuth=azimuth)
        # images = image_slices.copy()
        for i in range(0, len(labels)):
            # Balance validation because is used for choosing best model
            balance = self._parse_balance(balance_dataset, len(images))
            patches = self.apply_sliding(images[i], labels[i], size=size, stride=stride, classification=classification)
            images[i], labels[i] = self._generator_to_list(patches)
            if balance[i]:
                images[i], labels[i] = self.balance_patches(images[i], labels[i])
            elif classification and i < len(labels) - 1:
                images[i], _, labels[i], _ = sklearn.model_selection.train_test_split(images[i], labels[i],
                                                                                      train_size=0.2)
        if shuffle:  # No need to shuffle the rest as val and test does not really matter they are shuffled
            images[0], labels[0] = sklearn.utils.shuffle(images[0], labels[0])
        return images, labels

    def _get_single_image_separated_dataset(self, percentage: tuple, savefig: Optional[str] = None,
                                            balance_dataset: Union[bool, Tuple[bool]] = False,
                                            azimuth: Optional[str] = None, pad: bool = False):
        x, y = self._slice_dataset(percentage=percentage, savefig=savefig, azimuth=azimuth)
        balance = self._parse_balance(balance_dataset, length=len(y))
        for i in range(0, len(y)):
            if pad:
                x[i], y[i] = self._pad_image(x[i], y[i])
            x[i] = np.expand_dims(x[i], axis=0)
            y[i] = np.expand_dims(y[i], axis=0)
            if balance[i]:
                y[i] = self._balance_total_pixels_of_patch(y[i])
        return x, y

    # sparse <-> categorical (one-hot-encoded)
    @staticmethod
    def get_sparse_with_nul_label(one_hot_labels):
        sparse_labels = np.argmax(one_hot_labels, axis=-1) + 1
        # TODO: Isn't this better with np.where? Pseudo code:
        # sparse_labels = np.zeros(shape=one_hot_labels.shape[:2])
        # for x, y, z in np.where(one_hot_labels == 1):
        #   sparse_labels[x][y] = z + 1
        mask = np.all(one_hot_labels[:, :] == one_hot_labels.shape[-1] * [0.], axis=-1)
        sparse_labels[mask] = 0.
        return sparse_labels

    @staticmethod
    def sparse_to_categorical_2D(labels) -> np.ndarray:
        classes = np.max(labels)
        ground_truth = np.zeros(labels.shape + (classes,), dtype=float)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != 0:
                    ground_truth[i, j, labels[i, j] - 1] = 1.
        return ground_truth

    # Parser/check input
    @staticmethod
    def _parse_percentage(percentage) -> List[float]:
        if isinstance(percentage, int):
            assert percentage == 1
            percentage = (1.,)
        if isinstance(percentage, float):
            if 0 < percentage <= 1:
                percentage = (percentage,)
            else:
                raise ValueError(f"Percentage must be 0 < percentage <= 1, received {percentage}")
        else:
            percentage = list(percentage)
            assert all(p >= 0. for p in percentage), f"percentage elements can't be negative. Received {percentage}."
            assert sum(percentage) <= 1., f"percentage must add to 1 max, " \
                                          f"but it adds to sum({percentage}) = {sum(percentage)}"
        return percentage

    @staticmethod
    def _parse_pad(pad: PAD_TYPING, kernel_size: Union[int, Sequence]):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = tuple(kernel_size)
            assert len(kernel_size) == 2
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

    @staticmethod
    def _parse_balance(balance_dataset: Union[bool, Tuple[bool]], length: int) -> Tuple[bool]:
        if isinstance(balance_dataset, bool):
            balance = np.full(shape=length, fill_value=balance_dataset)
        elif len(balance_dataset) == length:
            balance = balance_dataset
        elif len(balance_dataset) < length:
            balance = tuple(balance_dataset) + (False,) * (length - len(balance_dataset))
        else:
            raise ValueError(f"Balance dataset ({balance_dataset}) was longer than expected (length = {length}).")
        assert np.all(isinstance(bal, bool) for bal in balance)
        return tuple(balance)

    # Methods to print rgb image

    def _diag_to_rgb(self, diagonal):
        # diagonal = coh_matrix[:, :, :3].astype(float)
        assert np.all(np.imag(diagonal) == 0), "ERROR: Imaginary part was not zero"
        assert diagonal.shape[-1] == 3, f"Unknown shape for diagonal ({diagonal.shape}). " \
                                        f"Expected last dimension to be 3"
        diagonal = diagonal.astype(float)
        diagonal[diagonal == 0] = float("nan")
        diag_db = 10 * np.log10(diagonal)
        noramlized_diag = np.zeros(shape=diag_db.shape)
        noramlized_diag[:, :, 0] = self.normalize_without_outliers(diag_db[:, :, 0])
        noramlized_diag[:, :, 2] = self.normalize_without_outliers(diag_db[:, :, 1])
        noramlized_diag[:, :, 1] = self.normalize_without_outliers(diag_db[:, :, 2])
        return np.nan_to_num(noramlized_diag)

    @staticmethod
    def remove_outliers(data, iqr=(1, 99)):
        low = np.nanpercentile(data, iqr[0])
        high = np.nanpercentile(data, iqr[1])
        return low, high

    def normalize_without_outliers(self, data):
        low, high = self.remove_outliers(data)
        return (data - low) / (high - low)

    # Methods with Tensorflow
    def _transform_to_tensor(self, x, y, batch_size: int, data_augment: bool = False, shuffle=True,
                             complex_mode: bool = True, real_mode: str = "real_imag"):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=BUFFER_SIZE)
        ds = ds.batch(batch_size)
        if data_augment:
            ds = ds.map(flip)
        if not complex_mode:
            ds = ds.map(lambda img, labels: transform_to_real_map_function(img, labels, real_mode))
        return ds

    # Balance dataset
    @staticmethod
    def balanced_test_split(x_all: List[Tuple[int, int]], y_all: np.ndarray, test_size, shuffle: bool) \
            -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray]:
        y_all = np.array(y_all)
        sparse_y = np.argmax(y_all, axis=-1)
        mask = np.full(shape=sparse_y.shape, fill_value=False)
        # train_size_totals = []
        for cls in range(y_all.shape[-1]):
            expected_size = int((1 - test_size) * y_all.shape[0] / y_all.shape[-1])
            min_size = y_all[sparse_y == cls].shape[0] - 1
            train_size = min(min_size, expected_size)
            if train_size == min_size:
                if train_size == 0:
                    raise ValueError(f"Train_size was 0. You run out of labels for class {cls}.")
                    # raise ValueError(f"Train_size was 0. You run out of labels for class {cls}")
                logging.warning(f"All samples ({train_size}) but one used for class {cls}. "
                                f"It was expected to have at least {expected_size} samples."
                                f"Try using a lower train percentage.")
            indexes = np.random.choice(np.where(sparse_y == cls)[0], size=train_size, replace=False)
            # train_size_totals.append(train_size)
            mask[indexes] = True
            assert len(indexes) == train_size
            # assert sum(mask) == sum(train_size_totals)
        inverted_mask = np.invert(mask)
        x_train = list(np.array(x_all)[mask])       # Cannot slice lists, so this is kind of nasty.
        y_train = y_all[mask]
        x_test = list(np.array(x_all)[inverted_mask])
        y_test = y_all[inverted_mask]
        if shuffle:
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        return x_train, x_test, y_train, y_test

    def balance_patches(self, patches: List[Tuple[int, int]], label_patches: List) \
            -> Tuple[List[Tuple[int, int]], np.ndarray]:
        label_patches = np.array(label_patches)
        if len(label_patches.shape) == 4:
            # self._sanity_check_total_one_class_images(label_patches)
            # labels_stats(label_patches)
            # First make 'one-class' images to be the same amount
            patches, label_patches = self._remove_exceeding_one_class_images(patches, label_patches)
            # Then make all classes pixels to be the same amount
            # labels_stats(label_patches)
            if __debug__:
                self._sanity_check_total_one_class_images(label_patches)
            # This was added into previous function for optimization
            label_patches = self._balance_total_pixels_of_patch(label_patches)
            # labels_stats(label_patches)
            # assert len(patches) == len(label_patches)
        elif len(label_patches.shape) == 2:
            patches, label_patches = self._balance_classification_patches(patches, label_patches)
        else:
            ValueError(f"Unknown shape for label_patches {label_patches.shape}")
        return patches, label_patches

    # Classification balance
    @staticmethod
    def _balance_classification_patches(patches: List[Tuple[int, int]], label_patches: Union[List, np.ndarray]) \
            -> Tuple[List[Tuple[int, int]], np.ndarray]:
        find_classes = np.where(label_patches == 1)
        assert len(label_patches) == len(find_classes[0])  # There was no empty label
        total_per_class = np.bincount(find_classes[-1])
        total_to_keep = np.min(total_per_class)
        if not total_to_keep:
            logging.warning(f"Class/es {np.where(total_per_class == 0)[0]} had no occurrences, "
                            f"balance will mean to remove everything. Pretending they don't exist and balance "
                            f"with the next non-zero class occurrences.")
            total_to_keep = np.min(total_per_class[np.nonzero(total_per_class)])
        mask_indexes = set()
        for cls in set(find_classes[-1]):
            indexes = find_classes[0][find_classes[-1] == cls]
            mask_indexes = mask_indexes.union(set(np.random.choice(indexes, size=total_to_keep, replace=False)))
        assert total_to_keep * len(set(find_classes[-1])) == len(mask_indexes)
        mask = [i in mask_indexes for i in range(len(patches))]
        patches = list(compress(patches, mask))  # Apply mask
        label_patches = np.array(list(compress(label_patches, mask)))
        if __debug__:
            counts = np.bincount(np.where(label_patches == 1)[-1])
            assert np.all(counts[np.nonzero(counts)] == counts[np.nonzero(counts)][0])
        return patches, label_patches

    # Segmentation Balance
    def _remove_exceeding_one_class_images(self, patches: List[Tuple[int, int]], label_patches: np.ndarray) \
            -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        This code receives labels and 2 cases are possible
            - Either the image has only one class present (together with unlabeled pixels)
            - Multiple-Labels
            - No labels will raise an error.
        The returned patches will have both
            1. same amount of total pixels per class and
            2. same amount of 'one-class-only' images
        :param patches: Images of shape (P, H, W, C)
        :param label_patches: one-hot-encoded labels of shape (P, H, W, cls)
        :return: tuple of balanced (patches, label_patches).
        """
        counter, indexes_to_keep, full_img_occurrences, \
        mixed_img_occurrences, total_img_occurrences = self._get_patch_image_counter_information(label_patches)
        min_images_occ = np.min(np.array(total_img_occurrences)[np.nonzero(total_img_occurrences)[0]])
        for cls in range(label_patches.shape[-1]):  # This loop removed the actual exceding one-class images
            if not total_img_occurrences[cls]:
                # Shall I let the user do this? or is obviously an error and raise one?
                logging.warning(f"Class {cls} has no labels present. Will be ignored.")
                continue
                # raise ValueError(f"Class {cls} has no labels present.")
            location_of_patches_for_given_class = counter[cls]["full"]  # This was ordered
            one_class_occ = max(min_images_occ - mixed_img_occurrences[cls], 0)
            # If min_class_occ == 0 I should just remove everything. So to_keep should be an empty list
            # As it is ordered, I keep the images that have the more occurences.
            to_keep = location_of_patches_for_given_class[-one_class_occ:] if one_class_occ else []
            counter[cls]["full"] = to_keep
            indexes_to_keep = indexes_to_keep.union(set([keep["index"] for keep in to_keep]))  # Saved the indexes saved
        mask = [i in indexes_to_keep for i in range(len(patches))]  # Keep saved indexes
        patches = list(compress(patches, mask))         # Apply mask
        label_patches = np.array(list(compress(label_patches, mask)))
        return patches, label_patches

    def _sanity_check_total_one_class_images(self, label_patches):
        _, _, _, mixed_img_occurrences, total_img_occurrences = self._get_patch_image_counter_information(label_patches)
        min_images_occ = np.min(np.array(total_img_occurrences)[np.nonzero(total_img_occurrences)[0]])
        counter = self._get_balanced_patch_image_counter_information(label_patches)
        pixel_occ_total = np.bincount(np.where(label_patches == 1)[-1])
        for cls in range(len(counter)):  # Sanity checks
            assert len(counter[cls]) == min_images_occ or len(counter[cls]) == mixed_img_occurrences[cls], \
                f"Total images of class {cls} should have been " \
                f"{max(mixed_img_occurrences[cls], min_images_occ)} but was {len(counter[cls])}"
            assert pixel_occ_total[cls] == sum([co["occurrences"] for co in counter[cls]])  # Occurrences sum make sense
            indexes = [co["index"] for co in counter[cls]]
            mask = [indx in indexes for indx in range(len(label_patches))]
            assert pixel_occ_total[cls] == sum(np.where(label_patches[mask] == 1)[-1] == cls)

    def _balance_total_pixels_of_patch(self, label_patches: np.ndarray) -> np.ndarray:
        counter = self._get_balanced_patch_image_counter_information(label_patches)
        pixel_occ_total = np.bincount(np.where(label_patches == 1)[-1])
        total_to_be_achieved = np.min(pixel_occ_total[np.nonzero(pixel_occ_total)])
        total_img_occurrences = []
        for cls in range(label_patches.shape[-1]):
            total_img_occurrences.append(len(counter[cls]))
        for cls in range(label_patches.shape[-1]):
            if not total_img_occurrences[cls]:
                continue
            if total_to_be_achieved < len(counter[cls]):
                logging.warning(
                    f"Desired size {total_to_be_achieved} for total pixels of one-class images for label {cls} "
                    f"is less than the total amount of images. Resulting that some images will be empty.\n"
                    f"setting a minimum value of {len(counter[cls])} "
                    f"to avoid this (one pixel label per image)")
                to_be_achieved = len(counter[cls])
            else:
                to_be_achieved = total_to_be_achieved
            logging.debug(f"Balancing class {cls} with a total of {total_img_occurrences[cls]} images and "
                          f"{pixel_occ_total[cls]} pixels and {to_be_achieved} to be achieved")
            # TODO: May be optimized if I do it for all classes at once
            label_patches = self._get_total_pixels_to_meet(label_patches, cls=cls, cls_counter=counter[cls],
                                                           to_be_achieved=to_be_achieved)
        if __debug__:
            class_occurrences = np.bincount(np.where(label_patches == 1)[-1])
            assert np.all(np.logical_or(class_occurrences == class_occurrences[np.nonzero(class_occurrences)][0],
                                        class_occurrences == 0))  # I allow zero in this case. Should I?
        # assert np.all([len(counter[i]) for i in range(1, len(counter))] == len(counter[1]))
        return label_patches

    def _get_total_pixels_to_meet(self, label_patches, cls, cls_counter, to_be_achieved):
        """
        Removes pixel classes randomly from images so that the total pixels (sum) of all images are gives to_be_achieved
        :param label_patches: patches of images labels
        :param cls_counter: list of tuples ("index", "occurrences") where index is the indexes from label_patches of
            the images to be filtered and occurrences the number of pixels that the image has
        :param to_be_achieved: Total sum of pixel classes to be met
        :return: balanced label_patches
        """
        if __debug__:
            backup_total = to_be_achieved
        total_images_to_be_used = len(cls_counter)
        avg = int(to_be_achieved / total_images_to_be_used)
        for i in range(len(cls_counter)):
            case = cls_counter[i]
            if case["occurrences"] <= avg:
                logging.debug(f"Image had {case['occurrences']} whereas needed avg was {avg}")
                to_be_achieved -= case["occurrences"]
            else:
                logging.debug(f"Removing {case['occurrences'] - avg} of a total of {case['occurrences']} "
                              f"to reach {avg}")
                assert len(np.where(np.where(label_patches[case["index"]] == 1)[-1] == cls)[-1]) == case["occurrences"]
                label_patches[case["index"]] = self._randomly_remove(image=label_patches[case["index"]], cls=cls,
                                                                     number_of_pixels=case["occurrences"] - avg)
                assert len(np.where(np.where(label_patches[case["index"]] == 1)[-1] == cls)[-1]) == avg
                to_be_achieved -= avg
            total_images_to_be_used -= 1
            if total_images_to_be_used:
                avg = int(to_be_achieved / total_images_to_be_used)
        if __debug__:
            achieved = 0
            for i in range(len(cls_counter)):
                case = cls_counter[i]
                achieved += len(np.where(np.where(label_patches[case["index"]] == 1)[-1] == cls)[-1])
            assert backup_total == achieved
        return label_patches

    @staticmethod
    def _randomly_remove(image, cls, number_of_pixels):
        """
        From an image of shape (H, W, cls), it removes a total amount of number_of_pixels setting them to [0, ..., 0]
            The image can have already empty labels, they will be contemplated.
        """
        available_x, available_y, available_z = np.where(image == 1)
        available_pixel = [(x, y) for x, y, z in zip(available_x, available_y, available_z) if z == cls]
        if len(available_pixel) < number_of_pixels:
            raise ValueError("Asked to remove more pixels than available in image")
        indexes = np.random.choice(range(len(available_pixel)), size=number_of_pixels, replace=False)
        for index in indexes:
            image[available_pixel[index][0]][available_pixel[index][1]] = np.array([0.] * image.shape[-1])
        return image

    # Counter generators for balancing
    @staticmethod
    def _get_patch_image_counter_information(label_patches):
        # counter is a Dictionary of the following format:
        # {"class_number": List[(
        #           "index": index of patch that contains the class
        #           "occurrences": number of pixels of the class for current image in the patch
        # )]}
        indexes_to_keep = set()  # Indexes of all patches images to keep
        counter = defaultdict(lambda: {"full": [], "mixed": []})
        for i, la in enumerate(label_patches):  # For each patch image.
            present_classes = np.where(la == 1)[-1]  # Find all classes (again, there will be at least one).
            if len(present_classes) == 0:
                raise IndexError(f"No patch image should have had an empty label.")
            all_equal = np.all(present_classes == present_classes[0])  # Are all classes the same one?
            for cls in set(present_classes):
                ordered_insertion_by_key(counter[cls]["full" if all_equal else "mixed"],
                                         # +1 because class = 0 is reserved for mix classes cases
                                         {"index": i, "occurrences": len(present_classes[present_classes == cls])},
                                         key="occurrences")
                if not all_equal:
                    indexes_to_keep.add(i)  # Keep all mixed figures, this is to make it faster
                    # If mixed case, then it should have been balanced in the sliding window operation so assert it?
                    # assert np.all([np.sum(present_classes == cls) for cls in set(present_classes)] ==
                    #               np.sum(present_classes == present_classes[0]))  # This was supposedly done before.
        full_img_occurrences, mixed_img_occurrences, total_img_occurrences = [], [], []
        for i in range(label_patches.shape[-1]):  # I use shape to make sure I go through every class
            full_img_occurrences.append(len(counter[i]["full"]))
            mixed_img_occurrences.append(len(counter[i]["mixed"]))
            total_img_occurrences.append(len(counter[i]["full"]) + len(counter[i]["mixed"]))
        return counter, indexes_to_keep, full_img_occurrences, mixed_img_occurrences, total_img_occurrences

    @staticmethod
    def _get_balanced_patch_image_counter_information(label_patches):
        counter = defaultdict(list)
        for i, la in enumerate(label_patches):  # For each patch image.
            present_classes = np.where(la == 1)[-1]  # Find all classes (again, there will be at least one).
            if len(present_classes) == 0:
                raise IndexError(f"No patch image should have had an empty label.")
            for cls in set(present_classes):
                ordered_insertion_by_key(counter[cls],
                                         # +1 because class = 0 is reserved for mix classes cases
                                         {"index": i, "occurrences": len(present_classes[present_classes == cls])},
                                         key="occurrences")
        return counter

    # MISC
    @staticmethod
    def _generator_to_list(patches):
        img_patches = []
        label_patches = []
        for elem in patches:
            img_patches.append(elem[0])
            label_patches.append(elem[1])
        return img_patches, label_patches

    def _slice_dataset(self, percentage: tuple, azimuth: Optional[str], savefig: Optional[str]):
        if azimuth is None:
            azimuth = self.azimuth
        if azimuth is None:
            raise ValueError("Azimuth direction was not defined.")
        azimuth = azimuth.lower()
        percentage = self._parse_percentage(percentage)
        if azimuth == "horizontal":
            total_length = self.image.shape[1]
        elif azimuth == "vertical":
            total_length = self.image.shape[0]
        else:
            raise ValueError(f"Azimuth {azimuth} unknown.")
        th = 0
        x_slice = []
        y_slice = []
        mask_slice = []
        for per in percentage:
            slice_1 = slice(th, th + int(total_length * per))
            th += int(total_length * per)
            if azimuth == "horizontal":
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
                self.print_ground_truth(label=y, transparent_image=x_slice[i],
                                        mask=mask_slice[i], path=str(savefig) + slices_names[i])
        return x_slice, y_slice

    @staticmethod
    def _pad_image(image, labels):
        first_dim_pad = int(2 ** 5 * np.ceil(image.shape[0] / 2 ** 5)) - image.shape[0]
        second_dim_pad = int(2 ** 5 * np.ceil(image.shape[1] / 2 ** 5)) - image.shape[1]
        paddings = [
            [int(np.ceil(first_dim_pad / 2)), int(np.floor(first_dim_pad / 2))],
            [int(np.ceil(second_dim_pad / 2)), int(np.floor(second_dim_pad / 2))],
            [0, 0]
        ]
        image = np.pad(image, paddings)
        labels = np.pad(labels, paddings)
        return image, labels

    def _separate_dataset(self, patches: Generator,
                          percentage: Union[Tuple[float], float], shuffle: bool = True, classification: bool = False,
                          balance_dataset: bool = False) -> Tuple[List[List[Tuple[int, int]]], List]:
        """
        Separates dataset patches according to the percentage
        :param patches: Generator of Tuple[ image patches, label_patches ]
        :param percentage: list of percentages for each value,
            example [0.9, 0.02, 0.08] to get 90% train, 2% val and 8% test
        :param shuffle: Shuffle dataset before split
        :param classification:
        :param balance_dataset:
        :return: tuple of lists of size = len(percentage), one with data x and other with labels y.
        """
        percentage = self._parse_percentage(percentage)
        balance = self._parse_balance(balance_dataset, length=len(percentage))
        x_test, y_test = self._generator_to_list(patches)
        x = []
        y = []
        full_percentage = np.isclose(sum(percentage), 1)  # Not the same to have (0.1, 0.2, 0.7) or (0.1, 0.2)
        if full_percentage:
            percentage = percentage[:-1]
        for i, per in enumerate(percentage):  # But this loop is the same
            if classification and balance[i]:
                x_train, x_test, y_train, y_test = self.balanced_test_split(x_test, y_test, test_size=1 - per,
                                                                            shuffle=shuffle)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=max(1 - per, 0.),
                                                                    shuffle=True if classification else shuffle,
                                                                    stratify=y_test if classification else None)
                if balance[i]:  # Balance all but last. This will balance train and val but not test
                    x_train, y_train = self.balance_patches(x_train, y_train)
            if i < len(percentage) - 1:
                percentage[i + 1:] = [value / (1 - percentage[i]) for value in percentage[i + 1:]]
            x.append(x_train.copy())
            y.append(y_train.copy())
        if full_percentage:
            if balance[-1]:
                x_test, y_test = self.balance_patches(x_test, y_test)
            x.append(x_test.copy())
            y.append(y_test.copy())
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

    def _sliding_window_operation(self, im, lab, size: Tuple[int, int], stride: int,
                                  pad: Tuple[Tuple[int, int], Tuple[int, int]],
                                  segmentation: bool = True, add_unlabeled: bool = False) \
            -> Tuple[List[Tuple[int, int]], List]:
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
        # tiles = []
        # label_tiles = []
        im = np.pad(im, (pad[0], pad[1], (0, 0)))
        lab = np.pad(lab, (pad[0], pad[1], (0, 0)))
        assert im.shape[0] > size[0] and im.shape[1] > size[1], f"Image shape ({im.shape[0]}x{im.shape[1]}) " \
                                                                f"is smaller than the window to apply " \
                                                                f"({size[0]}x{size[1]})"
        for x in range(0, im.shape[0] - size[0] + 1, stride):
            for y in range(0, im.shape[1] - size[1] + 1, stride):
                label_to_add = self.get_image_around_point(lab, x, y, size if segmentation else (1, 1), squeeze=True)
                image_to_add = self.get_image_around_point(im, x, y, size, squeeze=False)
                if add_unlabeled or (segmentation and not np.all(np.all(label_to_add == 0, axis=-1))) or \
                        (not segmentation and not np.all(label_to_add == 0)):
                    # label_tiles.append(label_to_add)
                    # tiles.append((x, y))
                    yield image_to_add, label_to_add
        # # Sanity checks
        # assert len(tiles) == len(label_tiles)
        # # assert np.all([p.shape == (size[0], size[1], im.shape[2]) for p in tiles])  # Commented, expensive assertion
        # assert add_unlabeled or np.all([not np.all(np.all(la == 0, axis=-1)) for la in label_tiles])
        # if not pad:  # If not pad then use equation 7 of https://www.mdpi.com/2072-4292/10/12/1984
        #     assert int(np.shape(tiles)[0]) == int(
        #         (np.floor((im.shape[0] - size[0]) / stride) + 1) * (np.floor((im.shape[1] - size[1]) / stride) + 1))
        # # tiles = np.array(tiles)
        # # label_tiles = np.array(label_tiles)
        # return tiles, label_tiles

    @staticmethod
    def get_image_around_point(image_to_crop: np.ndarray, x: int, y: int, size: Tuple[int, int], squeeze: bool = False):
        slice_x = slice(x, x + size[0])
        slice_y = slice(y, y + size[1])
        cropped = image_to_crop[slice_x, slice_y]
        if squeeze:
            cropped = np.squeeze(cropped)
        return cropped

    def get_patches_image_from_point_and_self_image(self, patches_points: List[List[Tuple[int, int]]],
                                                    size: Union[int, Tuple[int, int]],
                                                    pad: Optional[PAD_TYPING] = "same",
                                                    ) -> List:
        return self.get_patches_image_from_points(patches_points=patches_points,
                                                  image_to_crop=self.image,
                                                  size=size, pad=pad)

    def get_patches_image_from_points(self, patches_points: List[List[Tuple[int, int]]],
                                      image_to_crop,
                                      size: Union[int, Tuple[int, int]],
                                      pad: Optional[PAD_TYPING] = "same") -> List:
        """
        :param patches_points:
        :param image_to_crop: Either ND array. N >= 2. TODO: This constraint of N is not verified
            If first dimension equals patches_points first dimension it will use different images for each set
        :param size:
        :param pad:
        :return:
        """
        if isinstance(size, int):
            size = (size, size)
        else:
            size = tuple(size)
            assert len(size) == 2
        multiple_images = len(patches_points) == len(image_to_crop)
        if pad is not None:
            pad = self._parse_pad(pad, size)
            if not multiple_images:
                image_to_crop = np.pad(image_to_crop, (pad[0], pad[1], (0, 0)))
            else:
                image_to_crop = [np.pad(img, (pad[0], pad[1], (0, 0))) for img in image_to_crop]
        result = [np.empty(shape=(len(patches_points[i]),) + size + (image_to_crop[0].shape[-1],),
                           dtype=image_to_crop[i].dtype)
                  for i in range(len(patches_points))]
        for dset_index, dset in enumerate(patches_points):
            for points_index, points in enumerate(dset):
                result[dset_index][points_index] = self.get_image_around_point(image_to_crop=
                                                                               image_to_crop[dset_index] if
                                                                               multiple_images else image_to_crop,
                                                                               x=patches_points[dset_index][points_index][0],
                                                                               y=patches_points[dset_index][points_index][1],
                                                                               size=size)
        return result

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
        return np.transpose(k, axes=[1, 2, 0])

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

    """
        PUBLIC
    """

    def apply_sliding_on_self_data(self, *args, **kwargs) -> Tuple[List[Tuple[int, int]], List]:
        return self.apply_sliding(image=self.image, labels=self.labels, *args, **kwargs)

    def apply_sliding(self, image, labels, size: Union[int, Tuple[int, int]] = 128, stride: int = 25, pad="same",
                      classification: bool = False, add_unlabeled: bool = False) -> Tuple[List[Tuple[int, int]], List]:
        """
        Performs the sliding window operation to the image
        :param image:
        :param labels:
        :param size:
        :param stride:
        :param pad:
        :param classification:
        :param add_unlabeled: Add unlabeled data patch (False by default)
        :return: Tuple, first element is a list of the selected pixels (another tuple) and
            second element the labels of those pixels (Normally 3D)
        """
        if isinstance(size, int):
            size = (size, size)
        else:
            size = tuple(size)
            assert len(size) == 2
        pad = self._parse_pad(pad, size)
        # logging.debug(f"Computing swo on dataset {self.name}")
        # start = timeit.default_timer()
        return self._sliding_window_operation(image, labels, size=size, stride=stride, pad=pad,
                                              segmentation=not classification, add_unlabeled=add_unlabeled)
        # logging.debug(f"Computation done in {int((timeit.default_timer() - start) / 60)} minutes "
        #               f"{int(timeit.default_timer() - start)} seconds")
        # return patches, label_patches
