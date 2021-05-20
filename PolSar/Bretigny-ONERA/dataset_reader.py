import scipy.io
import numpy as np
import tensorflow as tf
from pdb import set_trace
from sklearn.model_selection import train_test_split
from cvnn.utils import randomize


def get_coherency_matrix(HH, VV, HV):
    # Section 2: https://earth.esa.int/documents/653194/656796/LN_Advanced_Concepts.pdf
    k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
    tf_k = tf.expand_dims(tf.transpose(k, perm=[1, 2, 0]), axis=-1)

    T = tf.linalg.matmul(tf_k, tf.transpose(tf_k, perm=[0, 1, 3, 2], conjugate=True))
    return T


def remove_unlabeled(x, y):
    """
    Removes the unlabeled pixels from both image and labels
    :param x: image input
    :param y: labels inputs. all values of 0's will be eliminated and its corresponging values of x
    :return: tuple (x, y) without the unlabeled pixels.
    """
    mask = y != 0
    return x[mask], y[mask]


def sparse_to_categorical_1D(labels):
    cat_labels = np.zeros((labels.shape[0], np.max(labels)))
    for i, val in enumerate(labels):
        if val != 0:
            cat_labels[i][val - 1] = 1.
    return cat_labels


def get_data(path: str = '/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'):
    mat = scipy.io.loadmat(path + '/bretigny_seg.mat')
    seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI.mat')
    # mat = scipy.io.loadmat(path + '/bretigny.mat')
    # seg = scipy.io.loadmat(path + '/bretigny_7ROI.mat')

    T = get_coherency_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    labels = seg['image']

    assert labels.shape[:2] == T.shape[:2]
    # Flatten labels and data
    flatten_T = tf.reshape(T, shape=(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3]))
    labels = np.reshape(labels, -1)
    # Remove unlabeled data
    flatten_T, labels = remove_unlabeled(flatten_T, labels)

    # Split train and test
    x_train, x_test, y_train, y_test = train_test_split(flatten_T.numpy(), labels, train_size=0.1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

    # Sparse into categorical labels
    y_test = sparse_to_categorical_1D(y_test)
    y_train = sparse_to_categorical_1D(y_train)
    y_val = sparse_to_categorical_1D(y_val)

    class_names = [c[0] for c in seg['name'].reshape(-1)]
    assert flatten_T.shape[1] == 9 and len(flatten_T.shape) == 2

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    T, labels = get_data()
    labels_values = set(np.reshape(labels, -1))
    set_trace()

