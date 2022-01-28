import os.path
import timeit
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
from cvnn.utils import standarize, transform_to_real_map_function

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
    [1, 0.349, 0.392],      # Red; Built-up Area
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

COLORS = {"BRET": BRET_COLORS, "OBER": OBER_COLORS, **SF_COLORS}

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


def pauli_rgb_map_plot(labels, dataset_name: str, t: Optional[np.ndarray] = None, path=None, mask=None, ax=None):
    labels_rgb = labels_to_rgb(labels, colors=COLORS[dataset_name], mask=mask)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    alpha = 1.
    # set_trace()
    # if t is not None:
    #     alpha = 0.4
    #     rgb = np.stack([t[:, :, 0], t[:, :, 1], t[:, :, 2]], axis=-1).astype(np.float32)
    #     ax.imshow(rgb)
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

    def __init__(self, root_path: str, name: str, mode: str, complex_mode: bool = True, real_mode: str = 'real_imag',
                 normalize: bool = False, balance_dataset: bool = False, classification: bool = False):
        self.root_path = Path(str(root_path))
        self.name = name
        assert mode.lower() in {"s", "t", "k"}
        self.mode = mode.lower()
        self.real_mode = real_mode.lower()
        self.complex_mode = complex_mode    # TODO: Best practice to leave it outside
        self.classification = classification
        self.balance_dataset = balance_dataset
        self.image = self.get_image()
        self.sparse_labels = self.get_sparse_labels()
        self.labels = self.get_labels()

    @abstractmethod
    def get_image(self) -> np.ndarray:
        pass

    def get_labels(self) -> np.ndarray:
        return self.sparse_to_categorical_2D(self.sparse_labels)

    @abstractmethod
    def get_sparse_labels(self) -> np.ndarray:
        pass

    def get_dataset(self, method: str, percentage: Union[Tuple[float], float] = 0.2,
                    size: int = 128, stride: int = 25, shuffle: bool = True, pad="same",
                    savefig: Optional[str] = None, orientation: str = "vertical", data_augment: bool = False,
                    remove_last: bool = False,
                    batch_size: int = cao_dataset_parameters['batch_size'], use_tf_dataset=False):
        if method == "random":
            x_patches, y_patches = self._get_shuffled_dataset(size=size, stride=stride, pad=pad, percentage=percentage,
                                                              shuffle=shuffle, remove_last=remove_last,
                                                              classification=self.classification)
        elif method == "separate":
            x_patches, y_patches = self._get_separated_dataset(percentage=percentage, size=size, stride=stride, pad=pad,
                                                               savefig=savefig, orientation=orientation,
                                                               shuffle=shuffle,
                                                               classification=self.classification)
        elif method == "single_separated_image":
            assert self.classification, f"Can't apply classification to the full image."
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
                ds_list = [transform_to_real_map_function(x, y, self.real_mode)
                           for i, (x, y) in enumerate(zip(x_patches, y_patches))]
        return tuple(ds_list)

    """
        PRIVATE
    """

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
            assert 0 < sum(percentage) <= 1., f"percentage must add to 1 max, " \
                                              f"but it adds to sum({percentage}) = {sum(percentage)}"
            if sum(percentage) < 1:
                percentage = percentage + [1 - sum(percentage)]
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

    @staticmethod
    def balanced_test_split(x_all, y_all, test_size, shuffle):
        x_train_per_class, x_test_per_class, y_train_per_class, y_test_per_class = [], [], [], []
        sparse_y = np.argmax(y_all, axis=-1)
        for cls in range(y_all.shape[-1]):
            x_train, x_test, y_train, y_test = train_test_split(x_all[sparse_y == cls], y_all[sparse_y == cls],
                                                                train_size=int(
                                                                    (1-test_size) * y_all.shape[0] / y_all.shape[-1]),
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

    def _separate_dataset(self, percentage: List[float], shuffle: bool = True,
                          remove_last: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param percentage: list of percentages for each value,
            example [0.9, 0.02, 0.08] to get 90% train, 2% val and 8% test.
        :param shuffle: Shuffle dataset before split.
        :return: tuple of two lists of size = len(percentage), one with data x and other with labels y.
        """
        percentage = self._parse_percentage(percentage)
        x_test = self.patches
        y_test = self.label_patches
        x = []
        y = []
        for i, per in enumerate(percentage[:-1]):
            if self.classification and self.balance_dataset:
                x_train, x_test, y_train, y_test = self.balanced_test_split(x_test, y_test, test_size=1 - per,
                                                                            shuffle=shuffle)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=1-per,
                                                                    shuffle=True if self.classification else shuffle,
                                                                    stratify=y_test if self.classification else None)
            percentage[i+1:] = [value / (1-percentage[i]) for value in percentage[i+1:]]
            x.append(x_train)
            y.append(y_train)
        if not remove_last:
            x.append(x_test)
            y.append(y_test)
        # print(f"memory before removing empty images usage {tf.config.experimental.get_memory_usage('GPU:0')} Bytes")
        x[0], y[0] = self._remove_empty_image(data=x[0], labels=y[0])
        # print(f"memory after removing empty images usage {tf.config.experimental.get_memory_usage('GPU:0')} Bytes")
        # set_trace()
        return x, y

    @staticmethod
    def _to_classification(x, y, mask=False):
        y = np.reshape(y[:, y.shape[1] // 2, y.shape[2] // 2, :], newshape=(y.shape[0], y.shape[-1]))
        # assert [np.all(y_patches_class[i][:] == y_patches[i][0][0][:]) for i in range(len(y_patches_class))]
        # 2. Remove empty pixels
        if mask:        # TODO: Remove this?
            mask = np.invert(np.all(y == 0, axis=-1))
            x = tf.boolean_mask(x, mask).numpy()
            y = tf.boolean_mask(y, mask).numpy()
        return x, y

    # Get dataset
    def _get_shuffled_dataset(self, size: int = 128, stride: int = 25,
                              percentage: Union[Tuple[float], float] = (0.8, 0.2),
                              shuffle: bool = True, pad="same", remove_last: bool = False,
                              classification: bool = False) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Applies the sliding window operations getting smaller images of a big image T.
        Splits dataset into train and test.
        :param size: Size of the window to be used on the sliding window operation.
        :param stride:
        :param percentage: float. Percentage of examples to be used for the test set [0, 1]
        :return: a Tuple of tf.Datasets (train_dataset, test_dataset)
        """
        self.apply_sliding(size=size, stride=stride, pad=pad, classification=classification)
        # del T, labels  # Free up memory
        x, y = self._separate_dataset(percentage=percentage, shuffle=shuffle, remove_last=remove_last)
        return x, y

    def _get_separated_dataset(self, percentage: tuple, size: int = 128, stride: int = 25, shuffle: bool = True, pad=0,
                               savefig: Optional[str] = None, orientation: str = "vertical", classification=False):
        images, labels = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)
        # train_slice_label = _balance_image(train_slice_label)
        if isinstance(size, int):
            size = (size, size)
        else:
            size = tuple(size)
            assert len(size) == 2
        for i in range(0, len(labels)):
            images[i], labels[i] = self._sliding_window_operation(images[i], labels[i],
                                                                  size=size, stride=stride,
                                                                  pad=self._parse_pad(pad, size))   # TODO: Use apply
            images[i], labels[i] = self._remove_empty_image(data=images[i], labels=labels[i])
            if classification:      # TODO: TEST THIS
                images[i], labels[i] = self._to_classification(x=images[i], y=labels[i])
            if shuffle:  # No need to shuffle the rest
                images[i], labels[i] = sklearn.utils.shuffle(images[i], labels[i])
        return images, labels

    def _get_single_image_separated_dataset(self, percentage: tuple, savefig: Optional[str] = None,
                                            orientation: str = "vertical", pad: bool = False):
        x, y = self._slice_dataset(percentage=percentage, savefig=savefig, orientation=orientation)
        for i in range(0, len(y)):
            if pad:
                x[i], y[i] = self._pad_image(x[i], y[i])
            x[i] = tf.expand_dims(x[i], axis=0)
            y[i] = tf.expand_dims(y[i], axis=0)
        return x, y

    @staticmethod
    def _remove_empty_image(data, labels):
        if len(labels.shape) == 4:
            mask = np.invert(np.all(np.all(labels == 0, axis=-1), axis=(1, 2)))
        elif len(labels.shape) == 2:
            mask = np.invert(np.all(labels == 0, axis=-1))
        else:
            raise ValueError(f"Ups, shape of labels of size {len(labels.shape)} not supported.")
        masked_filtered_data = tf.reshape(tf.boolean_mask(data, mask), shape=(-1,) + data.shape[1:])
        masked_filtered_labels = tf.boolean_mask(labels, mask)
        # filtered_data = []
        # filtered_labels = []
        # for i in range(0, len(labels)):
        #     if not np.all(labels[i] == 0):
        #         filtered_data.append(data[i])
        #         filtered_labels.append(labels[i])
        # filtered_data = np.array(filtered_data)
        # filtered_labels = np.array(filtered_labels)
        # assert np.all(masked_filtered_data.numpy() == filtered_data)
        # assert np.all(masked_filtered_labels.numpy() == filtered_labels)
        # set_trace()
        return masked_filtered_data, masked_filtered_labels

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

    def get_weights(self):
        return self._get_weights(self.get_labels())

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

    def apply_sliding(self, size: Union[int, Tuple[int, int]] = 128, stride: int = 25, pad="same",
                      classification: bool = False, remove_unlabeled: bool = False):
        if not hasattr(self, "patches"):
            if isinstance(size, int):
                size = (size, size)
            else:
                size = tuple(size)
                assert len(size) == 2
            pad = self._parse_pad(pad, size)
            temp_path = self.root_path / "dataset_preprocess_cache"
            os.makedirs(str(temp_path), exist_ok=True)
            config_string = f"{'cls' if classification else 'seg'}_{self.name.lower()}_window{size}_stride{stride}_pad{pad}"
            if os.path.isfile(temp_path / (config_string + "_patches.npy")):
                print(f"Loading dataset {config_string}_patches.npy")
                start = timeit.default_timer()
                self.patches = np.load(str(temp_path / (config_string + "_patches.npy")))
                self.label_patches = np.load(str(temp_path / (config_string + "_labels.npy")))
                print(f"Load done in {timeit.default_timer() - start} seconds")
            else:
                print(f"Computing dataset {config_string}_patches.npy")
                start = timeit.default_timer()
                self.patches, self.label_patches = self._sliding_window_operation(self.image, self.labels, size=size,
                                                                                  stride=stride, pad=pad)
                # print(f"patches shape after sliding window op{patches.shape}")
                if not os.path.exists(str(temp_path / ("seg" + config_string[3:] + "_patches.npy"))):
                    np.save(str(temp_path / ("seg" + config_string[3:] + "_patches.npy")), self.patches)
                    np.save(str(temp_path / ("seg" + config_string[3:] + "_labels.npy")), self.label_patches)
                if classification:
                    self.patches, self.label_patches = self._to_classification(x=self.patches, y=self.label_patches,
                                                                               mask=remove_unlabeled)
                    np.save(str(temp_path / ("cls" + config_string[3:] + "_patches.npy")), self.patches)
                    np.save(str(temp_path / ("cls" + config_string[3:] + "_labels.npy")), self.label_patches)
                print(f"Computation done in {timeit.default_timer() - start} seconds")
        else:
            print("Using previously computed values. Attention! if parameters have changed it will not take effect")
        # print(f"patches shape before going out of apply slifing {patches.shape}")
        return self.patches, self.label_patches

    # Utils
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
    def print_ground_truth(self, label: Optional = None, path=None, t=None, mask: Optional = None, ax=None):
        if label is None:
            label = self.get_labels()
        if mask is None:
            mask = self.get_sparse_labels()
        return pauli_rgb_map_plot(label, mask=mask, dataset_name=self.name, t=t if self.mode == "t" else None,
                                  path=path, ax=ax)
