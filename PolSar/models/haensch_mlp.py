import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy
from cvnn.layers import ComplexDense, ComplexFlatten, complex_input
from cvnn.losses import ComplexMeanSquareError      # TODO: There are 4 implementation in the paper of this

mlp_hyper_params = {
    'activation': 'cart_tanh',      # TODO: He actually talks of 2 options (polar as well)
    'shape': [10, 10],              # I just use the lowest error
    'optimizer': SGD(),
    'loss': ComplexMeanSquareError()
}

tf_mlp_hyper_params = {
    'activation': 'tanh',      #
    'loss': MeanSquaredError()
}


def _get_mlp_model(input_shape, num_classes, dtype, name='zhang_cnn'):
    in1 = complex_input(shape=input_shape, dtype=dtype)
    h = ComplexFlatten(dtype=dtype)(in1)
    for sh in mlp_hyper_params['shape']:
        h = ComplexDense(sh, activation=mlp_hyper_params['activation'], dtype=dtype)(h)
    out = ComplexDense(num_classes, activation='linear', dtype=dtype)(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=mlp_hyper_params['optimizer'], loss=mlp_hyper_params['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'), ComplexAverageAccuracy(name='average_accuracy')]
                  )
    return model


def _get_tf_mlp_model(input_shape, num_classes, dtype, name='zhang_cnn'):
    if dtype.is_complex:
        raise ValueError(f"Cannot use Tensorflow for creating a complex model")
    in1 = Input(shape=input_shape, dtype=dtype)
    h = Flatten(dtype=dtype)(in1)
    for sh in mlp_hyper_params['shape']:
        h = Dense(sh, activation=tf_mlp_hyper_params['activation'], dtype=dtype)(h)
    out = Dense(num_classes, activation='linear', dtype=dtype)(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=mlp_hyper_params['optimizer'], loss=tf_mlp_hyper_params['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'), ComplexAverageAccuracy(name='average_accuracy')]
                  )
    return model


def get_haensch_mlp_model(input_shape=(1, 1, 6), num_classes=6, dtype=np.complex64,
                          tensorflow: bool = False, name="my_model", dropout=None):
    if dropout is not None:
        raise ValueError("Dropout for zhang model not yet implemented")
    if not tensorflow:
        return _get_mlp_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                              name=name)
    else:
        return _get_tf_mlp_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                                 name=name)
