import scipy.io
import numpy as np
import tensorflow as tf
from pdb import set_trace


def get_coherency_matrix(HH, VV, HV):
    # Section 2: https://earth.esa.int/documents/653194/656796/LN_Advanced_Concepts.pdf
    k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
    tf_k = tf.expand_dims(tf.transpose(k, perm=[1, 2, 0]), axis=-1)

    T = tf.linalg.matmul(tf_k, tf.transpose(tf_k, perm=[0, 1, 3, 2], conjugate=True))
    return T


def get_data(path: str = '/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'):
    mat = scipy.io.loadmat(path + '/bretigny.mat')
    seg = scipy.io.loadmat(path + '/labels/seg.mat')

    T = get_coherency_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    labels = seg['seg']

    assert labels.shape[:2] == T.shape[:2]
    return T, labels


if __name__ == "__main__":
    T, labels = get_data()
    labels_values = set(np.reshape(labels, -1))
    set_trace()

