import scipy.io
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from pdb import set_trace
from typing import Tuple
from os import path
import sys
from sklearn.model_selection import train_test_split

from dataset_reader import PolsarDatasetHandler

sys.path.insert(1, '../')

if os.path.exists('/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'):
    path = '/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('D:/datasets/PolSar/Bretigny-ONERA/data'):
    path = 'D:/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'):
    path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'
elif os.path.exists('/home/cfren/Documents/onera/PolSar/Bretigny-ONERA/data'):
    path = '/home/cfren/Documents/onera/PolSar/Bretigny-ONERA/data'
else:
    raise FileNotFoundError("Dataset path not found")


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


class BretignyDataset(PolsarDatasetHandler):

    def __init__(self, mode: str, balanced: bool = False, *args, **kwargs):
        self.balanced = balanced
        super(BretignyDataset, self).__init__(root_path=path, name="BRET", mode=mode, *args, **kwargs)

    def print_ground_truth(self, t=None, *args, **kwargs):
        if t is None:
            t = self.get_image() if self.mode == "t" else None
        super(BretignyDataset, self).print_ground_truth(t=t, *args, **kwargs)

    def get_image(self) -> np.ndarray:
        if self.mode == "s":
            return self._get_bret_s_dataset()
        elif self.mode == "t":
            return self._get_bret_coherency_dataset()
        elif self.mode == "k":
            return self._get_bret_k_dataset()
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

    def get_sparse_labels(self):
        if not self.balanced:
            seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI.mat')
        else:
            seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI_balanced.mat')
        seg['image'] = seg['image'][:-3]
        return seg['image']

    """
        PRIVATE
    """

    def _open_data(self):
        mat = scipy.io.loadmat(path + '/bretigny_seg.mat')
        mat['HH'] = mat['HH'][:-3]
        mat['HV'] = mat['HV'][:-3]
        mat['VV'] = mat['VV'][:-3]
        return mat

    @staticmethod
    def _get_k_vector(HH, VV, HV):
        k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
        return tf.transpose(k, perm=[1, 2, 0])

    def _get_coherency_matrix(self, HH, VV, HV, kernel_shape=3):
        # Section 2: https://earth.esa.int/documents/653194/656796/LN_Advanced_Concepts.pdf
        k = self._get_k_vector(HH, VV, HV)
        tf_k = tf.expand_dims(k, axis=-1)  # From shape hxwx3 to hxwx3x1
        T = tf.linalg.matmul(tf_k, tf_k,
                             adjoint_b=True)  # k * k^H: inner 2 dimensions specify valid matrix multiplication dim
        one_channel_T = tf.reshape(T, shape=(T.shape[0], T.shape[1], T.shape[2] * T.shape[3]))  # hxwx3x3 to hxwx9
        removed_lower_part_T = _remove_lower_part(one_channel_T)  # hxwx9 to hxwx6 removing lower part of matrix
        filtered_T = mean_filter(removed_lower_part_T, kernel_shape)
        return filtered_T

    def _get_bret_coherency_dataset(self, kernel_shape=3):
        mat = self._open_data()
        T = self._get_coherency_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'], kernel_shape=kernel_shape)
        return T

    def _get_bret_k_dataset(self):
        mat = self._open_data()
        k = self._get_k_vector(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
        return k

    def _get_bret_s_dataset(self):
        mat = self._open_data()
        s = np.array([mat['HH'], mat['VV'], mat['HV']])
        s = tf.transpose(s, perm=[1, 2, 0])
        return s
