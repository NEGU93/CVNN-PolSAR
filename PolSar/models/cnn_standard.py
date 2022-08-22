import math

import numpy as np
import tensorflow as tf
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
    'padding': 'valid',
    'kernel_size':  3,
    'stride': 1,
    'complex_filters': [6, 12],
    'real_filters': [int(6*math.sqrt(2)), int(12*math.sqrt(2))],
    'pool_size': 2,
    'kernel_init': ComplexHeNormal(),
    'loss': ComplexAverageCrossEntropy(),       # End of II.A.4
    'activation': 'cart_relu',
    'optimizer': Adam
}


def _get_model(input_shape, num_classes, dtype, weights, name='cnn', learning_rate=0.0001):
    if dtype.is_complex:
        filters = "complex_filters"
    else:
        filters = "real_filters"
    in1 = complex_input(shape=input_shape, dtype=dtype)
    c1 = ComplexConv2D(filters=cnn_params_model[filters][0], kernel_size=cnn_params_model['kernel_size'],
                       strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                       kernel_initializer=cnn_params_model['kernel_init'],
                       activation='linear', dtype=dtype)(in1)
    # conv = ComplexBatchNormalization(dtype=dtype)(c1)
    conv = Activation(cnn_params_model['activation'])(c1)
    p1 = ComplexAvgPooling2D(pool_size=cnn_params_model['pool_size'], dtype=dtype)(conv)
    c2 = ComplexConv2D(filters=cnn_params_model[filters][1], kernel_size=cnn_params_model['kernel_size'],
                       strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                       kernel_initializer=cnn_params_model['kernel_init'],
                       activation='linear', dtype=dtype)(p1)
    # conv = ComplexBatchNormalization(dtype=dtype)(c2)
    conv = Activation(cnn_params_model['activation'])(c2)
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
    filters = "real_filters"
    in1 = Input(shape=input_shape)
    c1 = Conv2D(filters=cnn_params_model[filters][0], kernel_size=cnn_params_model['kernel_size'],
                strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                kernel_initializer=cnn_params_model['kernel_init'],
                activation='linear')(in1)
    # bn1 = BatchNormalization()(c1)
    a1 = Activation(cnn_params_model['activation'])(c1)
    p1 = AveragePooling2D(pool_size=cnn_params_model['pool_size'])(a1)
    c2 = Conv2D(filters=cnn_params_model[filters][1], kernel_size=cnn_params_model['kernel_size'],
                strides=cnn_params_model['stride'], padding=cnn_params_model['padding'],
                kernel_initializer=cnn_params_model['kernel_init'],
                activation='linear')(p1)
    # bn2 = BatchNormalization()(c2)
    a2 = Activation(cnn_params_model['activation'])(c2)
    flat = Flatten(dtype=dtype)(a2)
    out = Dense(num_classes, activation='softmax')(flat)
    model = Model(inputs=[in1], outputs=[out], name=name)
    if weights is not None:
        loss = ComplexWeightedAverageCrossEntropy(weights=1/weights)
    else:
        loss = cnn_params_model['loss']
    model.compile(optimizer=cnn_params_model['optimizer'](learning_rate=learning_rate, beta_1=0.9), loss=loss,
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy')
                           ])
    return model


def get_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 6), num_classes=15, dtype=np.complex64, weights=None,
                  tensorflow: bool = False, name="cnn", dropout=None, learning_rate=None):
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
