import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy
from cvnn.layers import ComplexDense, ComplexFlatten, complex_input, ComplexDropout
from cvnn.losses import ComplexAverageCrossEntropy
from cvnn.real_equiv_tools import get_real_equivalent_multiplier_from_shape

mlp_hyper_params = {
    'activation': 'cart_relu',
    'shape': [96, 180],
    'optimizer': Adam(learning_rate=0.0001, beta_1=0.9),
    'loss': ComplexAverageCrossEntropy()
}

tf_mlp_hyper_params = {
    'activation': 'relu',      #
    'loss': ComplexAverageCrossEntropy()
}


def _get_mlp_model(input_shape, num_classes, dtype, name='mlp', equiv_technique="ratio_tp"):
    in1 = complex_input(shape=input_shape, dtype=dtype)
    h = ComplexFlatten(dtype=dtype)(in1)
    shape = mlp_hyper_params['shape'].copy()
    if not dtype.is_complex:
        for i in range(len(mlp_hyper_params['shape'])):
            multiplier = get_real_equivalent_multiplier_from_shape([input_shape[-1]] + mlp_hyper_params['shape'] + [num_classes],
                                                                   equiv_technique=equiv_technique)
            shape[i] = int(np.ceil(shape[i] * multiplier[i]))
    for sh in shape:
        h = ComplexDense(sh, activation=mlp_hyper_params['activation'], dtype=dtype)(h)
        h = ComplexDropout(rate=0.5)(h)
    out = ComplexDense(num_classes, activation='cart_softmax', dtype=dtype)(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=mlp_hyper_params['optimizer'], loss=mlp_hyper_params['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'), ComplexAverageAccuracy(name='average_accuracy')]
                  )
    return model


def _get_tf_mlp_model(input_shape, num_classes, dtype, name='mlp', equiv_technique="ratio_tp"):
    if dtype.is_complex:
        raise ValueError(f"Cannot use Tensorflow for creating a complex model")
    in1 = Input(shape=input_shape, dtype=dtype)
    h = Flatten(dtype=dtype)(in1)
    shape = mlp_hyper_params['shape'].copy()
    for i in range(len(mlp_hyper_params['shape'])):
        multiplier = get_real_equivalent_multiplier_from_shape([input_shape[-1]] + mlp_hyper_params['shape'] + [num_classes],
                                                               equiv_technique=equiv_technique)
        shape[i] = int(np.round(shape[i] * multiplier[i]))
    for sh in shape:
        h = Dense(sh, activation=tf_mlp_hyper_params['activation'], dtype=dtype)(h)
        h = Dropout(rate=0.5)(h)
    out = Dense(num_classes, activation='softmax', dtype=dtype)(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=mlp_hyper_params['optimizer'], loss=tf_mlp_hyper_params['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'), ComplexAverageAccuracy(name='average_accuracy')]
                  )
    return model


def get_mlp_model(input_shape=(1, 1, 6), num_classes=6, dtype=np.complex64, tensorflow: bool = False, name="mlp",
                  dropout=None, equiv_technique="ratio_tp"):
    if dropout is not None:
        raise ValueError("Dropout for zhang model not yet implemented")
    if not tensorflow:
        return _get_mlp_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                              name=name, equiv_technique=equiv_technique)
    else:
        return _get_tf_mlp_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                                 name=name, equiv_technique=equiv_technique)
