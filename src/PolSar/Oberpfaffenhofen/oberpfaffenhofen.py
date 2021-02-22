import spectral.io.envi as envi
from pathlib import Path
from pdb import set_trace
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from itertools import compress
from cvnn.utils import standarize, randomize
from cvnn.dataset import Dataset
from cvnn.montecarlo import RealVsComplex
from cvnn import layers as complex_layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from typing import Optional
import tikzplotlib
import time


def separate_dataset(data, window_size: int = 9, stride: int = 3):
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


def open_dataset_t6():
    labels = scipy.io.loadmat('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat')['label']
    path = Path(
        '/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/Master_Track_Slave_Track/T6')
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


def remove_unlabeled(x, y):
    mask = y != 0
    return x[mask], y[mask]


def labels_to_ground_truth(labels):
    colors = np.array([
        [1, 0.349, 0.392],
        [0.086, 0.858, 0.576],
        [0.937, 0.917, 0.352]
    ])
    ground_truth = np.zeros(labels.shape + (3,), dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 0:
                ground_truth[i, j] = colors[labels[i, j] - 1]
    plt.imshow(ground_truth)
    plt.show()
    plt.imsave("ground_truth.pdf", ground_truth)
    tikzplotlib.save("ground_truth.tex")
    return ground_truth


def open_dataset_s2():
    path = Path('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/ESAR_Oberpfaffenhofen')
    labels = scipy.io.loadmat('/media/barrachina/data/datasets/PolSar/Oberpfaffenhofen/Label_Germany.mat')['label']

    # http://www.spectralpython.net/fileio.html#envi-headers
    s_11_meta = envi.open(path / 's11.bin.hdr', path / 's11.bin')
    s_12_meta = envi.open(path / 's12.bin.hdr', path / 's12.bin')
    s_21_meta = envi.open(path / 's21.bin.hdr', path / 's21.bin')
    s_22_meta = envi.open(path / 's22.bin.hdr', path / 's22.bin')

    s_11 = s_11_meta.read_band(0)
    s_12 = s_12_meta.read_band(0)
    s_21 = s_21_meta.read_band(0)
    s_22 = s_22_meta.read_band(0)

    return [s_11, s_12, s_21, s_22], labels


def separate_train_test(x, y, ratio=0.1):
    classes = set(y)
    x_ordered_database = []
    y_ordered_database = []
    for cls in classes:
        mask = y == cls
        x_ordered_database.append(x[mask])
        y_ordered_database.append(y[mask])
    len_train = int(y.shape[0]*ratio/len(classes))
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


def get_dataset():
    T, labels = open_dataset_t6()
    labels_to_ground_truth(labels)
    T, labels = remove_unlabeled(T, labels)
    labels -= 1  # map [1, 3] to [0, 2]
    T = T.reshape(-1, T.shape[-1])
    labels = labels.reshape(np.prod(labels.shape))
    return T, labels


def run_monte(dataset, validation_data, test_data, iterations=10, epochs=200, batch_size=100,
              optimizer='sgd', shape_raw=None, activation='cart_relu',
              polar=False, dropout: Optional[float] = 0.5):
    if shape_raw is None:
        shape_raw = [50]

    # Create complex network
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output
    if len(shape_raw) == 0:
        print("No hidden layers are used. activation and dropout will be ignored")
        shape = [
            complex_layers.InputLayer(input_shape=input_size, dtype=np.complex64),
            complex_layers.ComplexDense(units=output_size, activation='softmax_real', dtype=np.complex64)
        ]
    else:  # len(shape_raw) > 0:
        shape = [complex_layers.InputLayer(input_shape=input_size, dtype=np.complex64)]
        for i in range(0, len(shape_raw)):
            shape.append(complex_layers.ComplexDense(units=shape_raw[i], activation=activation))
            if dropout is not None:
                shape.append(complex_layers.ComplexDropout(rate=dropout))
        shape.append(complex_layers.ComplexDense(units=output_size, activation='softmax_real'))

    complex_network = Sequential(name="complex_network", layers=shape)
    complex_network.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

    monte_carlo = RealVsComplex(complex_network)
    for model in monte_carlo.models:
        model.summary()
    monte_carlo.output_config['plot_all'] = True
    monte_carlo.output_config['confusion_matrix'] = True
    monte_carlo.run(x=dataset.x, y=dataset.y, validation_data=validation_data, test_data=test_data,
                    data_summary=dataset.dataset_name, iterations=iterations, epochs=epochs, batch_size=batch_size, polar=polar)


if __name__ == '__main__':
    print("Loading Dataset")
    T, labels = get_dataset()
    T, labels = randomize(T, labels)
    x_train, y_train, x_test, y_test = separate_train_test(T, labels, ratio=0.1)
    x_train, y_train, x_val, y_val = separate_train_test(x_train, y_train, ratio=0.8)
    y_train = Dataset.sparse_into_categorical(y_train)
    y_test = Dataset.sparse_into_categorical(y_test)
    y_val = Dataset.sparse_into_categorical(y_val)
    dataset = Dataset(x_train.astype(np.complex64), y_train, dataset_name='Oberpfaffenhofen')
    print("Training model")
    run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
              test_data=(x_test.astype(np.complex64), y_test),
              iterations=50, epochs=300, batch_size=100,
              shape_raw=[100, 50], dropout=0.5, activation='cart_relu',
              polar=False)
    """
    shapes = [
        # [],
        # [5],
        [10], [50],
        [10, 10], [50, 50],
        [100, 50], [200, 100],
        [500], [500, 500],
        [1000]
        # [5000],
        # [500, 500], [5000, 5000],
        # [10000]
    ]
    for shape in shapes:
        run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
                  iterations=10, epochs=300,
                  optimizer='sgd', shape_raw=shape, activation='cart_relu', polar=False, dropout=None)"""
    """
    optimizers = [# 'sgd'
                  'rmsprop'
                  # cvnn.optimizers.SGD(learning_rate=0.1), cvnn.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  # cvnn.optimizers.SGD(learning_rate=0.01, momentum=0.75),
                  # cvnn.optimizers.RMSprop(learning_rate=0.001, rho=0.999, momentum=0.9),
                  # cvnn.optimizers.RMSprop(learning_rate=0.001, rho=0.8, momentum=0.75)]
                  ]
    functions = [# 'cart_relu',
                 'cart_tanh', 'cart_sigmoid']
    polar_modes = [
                   False
                   # True
    ]
    for opt in optimizers:
        for pol in polar_modes:
            for act_fun in functions:
                for shape in shapes:
                    run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
                              iterations=100, epochs=300,
                              optimizer=opt, shape_raw=shape, activation=act_fun, polar=pol)"""
