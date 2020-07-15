import numpy as np
import tensorflow as tf
import sys
import os
import pandas as pd
from pdb import set_trace

from cvnn.dataset import Dataset
from cvnn.utils import load_matlab_matrices
from cvnn.montecarlo import mlp_run_montecarlo
import keras
import complexnn
from datetime import datetime

montecarlo_options = {'CVNNvsRVNN', 'TFversion'}


def load_gilles_mat_data(fname, default_path="/media/barrachina/data/gilles_data/"):
    mat = load_matlab_matrices(fname, default_path)
    ic = mat['ic'].squeeze(axis=0)  # Labels corresponding to types
    ic = ic - 1  # Changed from matlab indexing to python (start with 0 and not with 1)
    nb_sig = mat['nb_sig'].squeeze(axis=0)  # number of examples for each label (label=position_ic)
    sx = mat['sx'][0]  # Unknown scalar
    types = [t[0] for t in mat['types'].squeeze(axis=0)]  # labels legends
    xp = []  # Metadata TODO: good for regression network
    for t in mat['xp'].squeeze(axis=1):
        xp.append({'Type': t[0][0], 'Nb_rec': t[1][0][0], 'Amplitude': t[2][0][0], 'f0': t[3][0][0],
                   'Bande': t[4][0][0], 'Retard': t[5][0][0], 'Retard2': t[6][0][0], 'Sequence': t[7][0][0]})

    xx = mat['xx'].squeeze(axis=2).squeeze(axis=1).transpose()  # Signal data

    return ic, nb_sig, sx, types, xp, xx


if __name__ == '__main__':
    data_2chirps = "data_cnn1d.mat"
    data_all_classes = "data_cnn1dC.mat"
    data_2chirps_test = "data_cnn1dT.mat"

    data_name = data_all_classes
    # bkeras_complex(data_name)
    # train_monte_carlo(data_name, iterations=1000, montecarlo='CVNNvsRVNN')

    # Show results
    # show_monte_carlo_results(data_name, True, montecarlo='CVNNvsRVNN')

    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = Dataset.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x = xx.astype(np.complex64)
    y = cat_ic.astype(np.float32)
    # x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)
    # train net
    dataset = Dataset(x=x, y=y)
    montecarlo_analyzer = mlp_run_montecarlo(dataset=dataset)


