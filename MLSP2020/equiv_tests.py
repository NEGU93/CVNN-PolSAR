from cvnn.montecarlo import run_gaussian_dataset_montecarlo
import cvnn.layers as layers
from cvnn.layers import Dense
from cvnn.cvnn_model import CvnnModel
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
import time
from typing import List, Optional


def run_with_shape(shape_raw: List, dropout: Optional[float] = 0.5):
    # Create complex network
    input_size = 128  # Size of input
    output_size = 2  # Size of output
    layers.ComplexLayer.last_layer_output_dtype = None
    layers.ComplexLayer.last_layer_output_size = None
    if len(shape_raw) == 0:
        print("No hidden layers are used. activation and dropout will be ignored")
        shape = [
            Dense(input_size=input_size, output_size=output_size, activation='softmax_real',
                  input_dtype=np.complex64, dropout=None)
        ]
    else:  # len(shape_raw) > 0:
        shape = [Dense(input_size=input_size, output_size=shape_raw[0], activation='cart_relu',
                       input_dtype=np.complex64, dropout=dropout)]
        for i in range(1, len(shape_raw)):
            shape.append(Dense(output_size=shape_raw[i], activation='cart_relu', dropout=dropout))
        shape.append(Dense(output_size=output_size, activation='softmax_real', dropout=None))

    complex_network = CvnnModel(name="complex_network", shape=shape, loss_fun=categorical_crossentropy,
                                optimizer='sgd', verbose=False, tensorboard=False)
    input_size = 2 * 128    # Size of input
    output_size = 2         # Size of output
    layers.ComplexLayer.last_layer_output_dtype = None
    layers.ComplexLayer.last_layer_output_size = None
    if len(shape_raw) == 0:
        print("No hidden layers are used. activation and dropout will be ignored")
        shape = [
            Dense(input_size=input_size, output_size=output_size, activation='softmax_real',
                  input_dtype=np.float32, dropout=None)
        ]
    else:  # len(shape_raw) > 0:
        shape = [Dense(input_size=input_size, output_size=shape_raw[0], activation='cart_relu',
                       input_dtype=np.float32, dropout=dropout)]
        for i in range(1, len(shape_raw)):
            shape.append(Dense(output_size=shape_raw[i], activation='cart_relu', dropout=dropout))
        shape.append(Dense(output_size=output_size, activation='softmax_real', dropout=None))

    real_network = CvnnModel(name="real_network", shape=shape, loss_fun=categorical_crossentropy,
                             optimizer='sgd', verbose=False, tensorboard=False)

    # add models
    models = []
    models.append(complex_network)
    time.sleep(1)
    models.append(complex_network.get_real_equivalent(capacity_equivalent=True, equiv_technique='ratio', name='ratio'))
    time.sleep(1)
    models.append(complex_network.get_real_equivalent(capacity_equivalent=False, name='double HL'))
    time.sleep(1)
    models.append(real_network)

    run_gaussian_dataset_montecarlo(models=models, iterations=100, dropout=0.5)


if __name__ == "__main__":
    print("1 HL")
    shapes = [8, 16, 32, 64, 128, 512, 1024]
    for sha in shapes:
        run_with_shape([sha])

    print("1 HL")
    shapes = [
        [32, 80],
        [128, 40],
        [512, 100],
        [1024, 512],
        [2048, 1024]
    ]
    for sha in shapes:
        run_with_shape(sha)
