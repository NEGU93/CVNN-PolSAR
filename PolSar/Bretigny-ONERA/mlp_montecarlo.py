import tensorflow as tf
import numpy as np
from pdb import set_trace
from cvnn.montecarlo import mlp_run_real_comparison_montecarlo, get_mlp, run_montecarlo
from cvnn.dataset import Dataset
from dataset_reader import get_coh_data

import cvnn.layers as layers
from cvnn.layers import ComplexDense, ComplexDropout


def test_activations():
    x_train, y_train, x_val, y_val = get_coh_data()
    dataset = Dataset(x=x_train, y=y_train)
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output

    models = []
    activations = ['zrelu', 'cart_relu', 'complex_cardioid', 'modrelu']
    for act in activations:
        models.append(get_mlp(input_size=input_size, output_size=output_size, activation=act, name=act))

    run_montecarlo(models=models, dataset=dataset,
                   iterations=10, epochs=150, batch_size=100, display_freq=1,
                   validation_data=(x_val, y_val),
                   debug=False, do_conf_mat=False, do_all=True, tensorboard=False, polar=None, plot_data=False)


def test_output_act():
    x_train, y_train, x_val, y_val = get_coh_data()
    dataset = Dataset(x=x_train, y=y_train)
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output

    models = []
    activations = ['softmax_real_with_abs', 'softmax_real_with_avg', 'softmax_real_with_mult',
                   'softmax_of_softmax_real_with_mult', 'softmax_of_softmax_real_with_avg', 'softmax_real_with_polar']
    for act in activations:
        models.append(get_mlp(input_size=input_size, output_size=output_size, output_activation=act, name=act))

    run_montecarlo(models=models, dataset=dataset,
                   iterations=10, epochs=150, batch_size=100, display_freq=1,
                   validation_data=(x_val, y_val),
                   debug=False, do_conf_mat=False, do_all=True, tensorboard=False, polar=None, plot_data=False)


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = get_coh_data()
    dataset = Dataset(x=x_train, y=y_train)

    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=30,
                                       epochs=150, batch_size=100, display_freq=1,
                                       optimizer='sgd', dropout=0.5, shape_raw=[100, 50], activation='cart_relu',
                                       polar=None,      # output_activation='softmax_real_with_avg',
                                       debug=False, do_all=True, shuffle=False, tensorboard=False, plot_data=False,
                                       validation_data=(x_val, y_val),
                                       capacity_equivalent=True, equiv_technique='ratio'
                                       )
