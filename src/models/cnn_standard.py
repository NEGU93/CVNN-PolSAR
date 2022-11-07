import math

import numpy as np
import tensorflow as tf
from typing import Optional, Dict
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, AveragePooling2D
from cvnn.layers import complex_input, ComplexConv2D, ComplexAvgPooling2D, ComplexFlatten, ComplexDense, \
    ComplexBatchNormalization
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy
from cvnn.losses import ComplexAverageCrossEntropy, ComplexWeightedAverageCrossEntropy
from cvnn.initializers import ComplexHeNormal, ComplexGlorotUniform
from cvnn.activations import cart_softmax


IMG_HEIGHT = 12
IMG_WIDTH = 12

cnn_params_model = {
    'padding': 'same',
    'kernel_size':  3,
    'stride': 1,
    'complex_filters': [6, 12],
    'pool_size': 2,
    'kernel_init': ComplexHeNormal(),
    'loss': ComplexAverageCrossEntropy(),       # End of II.A.4
    'activation': 'cart_relu',
    'optimizer': Adam
}


def _get_model(input_shape, num_classes, dtype, weights, name='cnn', learning_rate=0.0001):
    in1 = complex_input(shape=input_shape, dtype=dtype)
    conv = in1
    for i in range(len(cnn_params_model["complex_filters"])):
        filters = cnn_params_model["complex_filters"][i]
        conv = ComplexConv2D(filters=filters if dtype.is_complex else int(filters*math.sqrt(2)),
                             kernel_size=cnn_params_model['kernel_size'],
                             strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                             kernel_initializer=cnn_params_model['kernel_init'],
                             activation='linear', dtype=dtype)(conv)
        # conv = ComplexBatchNormalization(dtype=dtype)(c1)
        conv = Activation(cnn_params_model['activation'])(conv)
        if i < len(cnn_params_model["complex_filters"]) - 1:
            conv = ComplexAvgPooling2D(pool_size=cnn_params_model['pool_size'], dtype=dtype)(conv)
    flat = ComplexFlatten(dtype=dtype)(conv)
    out = ComplexDense(num_classes, activation='cart_softmax', dtype=dtype)(flat)
    model = Model(inputs=in1, outputs=out, name=name)

    if weights is not None:
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = cnn_params_model['loss']

    model.compile(optimizer=cnn_params_model['optimizer'](learning_rate=learning_rate, beta_1=0.9), loss=loss,
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy')])
    return model


def _get_tf_model(input_shape, num_classes, dtype, weights, name='tf_cnn', learning_rate=0.0001):
    if dtype.is_complex:
        raise ValueError(f"Cannot use Tensorflow for creating a complex model")
    in1 = Input(shape=input_shape)
    conv = in1
    for i in range(len(cnn_params_model["complex_filters"])):
        conv = Conv2D(filters=int(math.sqrt(2)*cnn_params_model["complex_filters"][i]),
                      kernel_size=cnn_params_model['kernel_size'],
                      strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                      kernel_initializer=cnn_params_model['kernel_init'], activation='linear')(conv)
        conv = Activation(cnn_params_model['activation'])(conv)
        if i < len(cnn_params_model["complex_filters"]) - 1:
            conv = AveragePooling2D(pool_size=cnn_params_model['pool_size'])(conv)
    flat = Flatten(dtype=dtype)(conv)
    out = Dense(num_classes, activation='softmax')(flat)
    model = Model(inputs=[in1], outputs=[out], name=name)
    if weights is not None:
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = cnn_params_model['loss']
    model.compile(optimizer=cnn_params_model['optimizer'](learning_rate=learning_rate, beta_1=0.9), loss=loss,
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy')
                           ])
    return model


def get_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 6), num_classes=15, dtype=np.complex64, weights=None,
                  tensorflow: bool = False, name="cnn", dropout=None, learning_rate=None,
                  hyper_dict: Optional[Dict] = None):
    if hyper_dict is not None:
        for key, value in hyper_dict.items():
            if key in cnn_params_model.keys():
                cnn_params_model[key] = value
            else:
                print(f"WARGNING: parameter {key} is not used")
    if dropout is not None:
        raise ValueError("Dropout for zhang model not yet implemented")
    if learning_rate is None:
        learning_rate = 0.0001
    if not tensorflow:
        return _get_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                          weights=weights, name=name, learning_rate=learning_rate)
    else:
        return _get_tf_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                             weights=weights, name=name, learning_rate=learning_rate)


if __name__ == '__main__':
    model = get_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 9))
    model.summary()
