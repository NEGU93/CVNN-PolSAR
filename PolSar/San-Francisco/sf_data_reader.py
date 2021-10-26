import numpy as np
from imageio import imread
from pathlib import Path
from os import path
from pdb import set_trace
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

if path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar')
elif path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from dataset_reader import sparse_to_categorical_2D, open_s_dataset, open_t_dataset_t3, labels_to_ground_truth, \
    get_dataset_for_cao_segmentation, SF_COLORS, get_single_image_separated_dataset

root_path = "/media/barrachina/data/datasets/PolSar/San Francisco/PolSF"
AVAILABLE_IMAGES = {
    "AIRSAR": {"x1": 0, "y1": 0, "x2": 1024, "y2": 900, "y_inverse": False},
    "ALOS2": {"x1": 736, "y1": 2832, "x2": 3520, "y2": 7888, "y_inverse": True},
    # "GF3": {"x1": 1144, "y1": 3464, "x2": 3448, "y2": 6376, "y_inverse": True},
    "RS2": {"x1": 661, "y1": 7326, "x2": 2041, "y2": 9126, "y_inverse": False},
    # "RISAT": {"x1": 2486, "y1": 4257, "x2": 7414, "y2": 10648, "y_inverse": False},   # RISAT is not Pol
}


def pauli_rgb_map_plot(t, labels, path=None, dataset_name: str = "AIRSAR", mask=None):
    labels_rgb = labels_to_ground_truth(labels, colors=SF_COLORS[dataset_name], mask=mask)
    rgb = np.stack([t[:, :, 0], t[:, :, 1], t[:, :, 2]], axis=-1).astype(np.float32)
    # for i in range(0, 5):
    #     std = tf.expand_dims(tf.expand_dims(tf.math.reduce_std(rgb, axis=[0, 1]), axis=0), axis=0)
    #     mean = tf.expand_dims(tf.expand_dims(tf.math.reduce_mean(rgb, axis=[0, 1]), axis=0), axis=0)
    #     rgb = tf.where(rgb > (mean + 1.*std), mean + std, rgb)
    #     rgb = tf.where(rgb < (mean - 1.*std), mean - std, rgb)
    # minimum = tf.expand_dims(tf.expand_dims(tf.reduce_min(rgb, axis=[0, 1]), axis=0), axis=0)
    # maximum = tf.expand_dims(tf.expand_dims(tf.reduce_max(rgb, axis=[0, 1]), axis=0), axis=0)
    # normalized_rgb = (rgb - minimum) / (maximum - minimum)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.imshow(labels_rgb, alpha=0.4)
    if path is not None:
        fig.savefig(path / "pauli_and_labels.png")
    else:
        plt.show()


def get_labels(open_data: str):
    folder = "SF-" + open_data
    labels = imread(Path(root_path) / folder / (folder + "-label2d.png"))
    return labels


def open_image(open_data: str = "AIRSAR", mode: str = "t", save_image: bool = False):
    open_data = open_data.upper()
    assert open_data in AVAILABLE_IMAGES, f"Unknown data {open_data}."
    folder = "SF-" + open_data
    sub_folder = "SAN_FRANCISCO_" + open_data
    labels = get_labels(open_data=open_data)
    one_hot_labels = sparse_to_categorical_2D(labels)
    mode = mode.lower()
    if mode == "s":
        data = open_s_dataset(str(Path(root_path) / folder / sub_folder))
    elif mode == "t":
        data = open_t_dataset_t3(str(Path(root_path) / folder / sub_folder / "T4"))
    else:
        raise ValueError(f"Mode {mode} not supported.")
    data = data[
           AVAILABLE_IMAGES[open_data]["y1"]:AVAILABLE_IMAGES[open_data]["y2"],
           AVAILABLE_IMAGES[open_data]["x1"]:AVAILABLE_IMAGES[open_data]["x2"]
           ]
    assert data.shape[:-1] == one_hot_labels.shape[:-1], f"dataset of shape {data.shape[:-1]} not corresponding with " \
                                                         f"labels of shape {one_hot_labels.shape[:-1]} for {open_data}"
    if AVAILABLE_IMAGES[open_data]["y_inverse"]:
        data = np.flip(data, axis=0)
    if save_image:
        pauli_rgb_map_plot(data, one_hot_labels, path=Path(root_path) / folder, dataset_name=open_data, mask=labels)
    return data, one_hot_labels


def get_sf_cao_segmentation(open_data: str = "AIRSAR", mode: str = "t",
                            complex_mode: bool = True, real_mode: str = 'real_imag'):
    img, label = open_image(open_data=open_data, mode=mode, save_image=False)
    train_dataset, test_dataset, weights = get_dataset_for_cao_segmentation(img, label, complex_mode=complex_mode,
                                                                            shuffle=True, mode=real_mode)
    del img, label
    return train_dataset, test_dataset, weights


def get_sf_separated(open_data: str = "AIRSAR", mode: str = "t",
    complex_mode: bool = True, real_mode: str = 'real_imag'):
    img, label = open_image(open_data=open_data, mode=mode, save_image=False)
    train_dataset, val_dataset, test_dataset, weights = get_single_image_separated_dataset(img, label,
                                                                                           percentage=(0.8, 0.1, 0.1),
                                                                                           complex_mode=complex_mode,
                                                                                           mode=real_mode,
                                                                                           orientation="vertical",
                                                                                           pad=True)
    del img, label
    return train_dataset, test_dataset, weights


if __name__ == "__main__":
    open_image(open_data="ALOS2", save_image=True)
