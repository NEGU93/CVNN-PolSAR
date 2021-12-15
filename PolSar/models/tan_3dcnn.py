import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Input, Conv3D, AveragePooling3D, Flatten
from cvnn.layers import complex_input, ComplexConv3D, ComplexAvgPooling3D, ComplexFlatten
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy

IMG_HEIGHT = 12
IMG_WIDTH = 12

INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 6)

tao_params_model = {
    'padding': 'same',                                      # Sec II
    'stride': 1,                                            # Needed to make numbers meet
    'kernel_size':  (3, 3, 3),                              # Fig 1 & Sec II
    'pool_size': (2, 2, 1),                                 # Fig 1 & Sec II
    'filters': [16, 16, 32, 32],                            # Fig 1 & Sec II
    'activation': 'cart_relu',                              # Sec II.C.2
    'tf_activation': 'relu',
    'optimizer': SGD(learning_rate=0.001, momentum=0.9),    # Start of II.B
    'loss': CategoricalCrossentropy(),                      # Sec II.C.5
}


def _get_model(num_classes, dtype, input_shape=INPUT_SHAPE, name='tan_3d_cnn'):
    in1 = complex_input(shape=input_shape, dtype=dtype)
    reshaped = tf.expand_dims(in1, axis=-1)
    c1 = ComplexConv3D(filters=tao_params_model['filters'][0], kernel_size=tao_params_model['kernel_size'],
                       strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                       activation=tao_params_model['activation'], dtype=dtype)(reshaped)
    c2 = ComplexConv3D(filters=tao_params_model['filters'][1], kernel_size=tao_params_model['kernel_size'],
                       strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                       activation=tao_params_model['activation'], dtype=dtype)(c1)
    p1 = ComplexAvgPooling3D(pool_size=tao_params_model['pool_size'], dtype=dtype)(c2)
    c3 = ComplexConv3D(filters=tao_params_model['filters'][2], kernel_size=tao_params_model['kernel_size'],
                       strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                       activation=tao_params_model['activation'], dtype=dtype)(p1)
    c4 = ComplexConv3D(filters=tao_params_model['filters'][3], kernel_size=tao_params_model['kernel_size'],
                       strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                       activation=tao_params_model['activation'], dtype=dtype)(c3)
    flat = ComplexFlatten(dtype=dtype)(c4)
    if dtype.is_complex:
        cast = tf.concat([tf.math.real(flat), tf.math.imag(flat)], axis=-1)
    else:
        cast = flat
    h = Dense(128, activation=tao_params_model['activation'], dtype=dtype.real_dtype)(cast)
    out = Dense(num_classes, activation='softmax', dtype=dtype.real_dtype)(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=tao_params_model['optimizer'], loss=tao_params_model['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy')])
    return model


def _get_tf_model(num_classes, input_shape=INPUT_SHAPE, name='tan_tf_3d_cnn'):
    in1 = Input(shape=input_shape)
    reshaped = tf.expand_dims(in1, axis=-1)
    c1 = Conv3D(filters=tao_params_model['filters'][0], kernel_size=tao_params_model['kernel_size'],
                strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                activation=tao_params_model['tf_activation'])(reshaped)
    c2 = Conv3D(filters=tao_params_model['filters'][1], kernel_size=tao_params_model['kernel_size'],
                strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                activation=tao_params_model['tf_activation'])(c1)
    p1 = AveragePooling3D(pool_size=tao_params_model['pool_size'])(c2)
    c3 = Conv3D(filters=tao_params_model['filters'][2], kernel_size=tao_params_model['kernel_size'],
                strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                activation=tao_params_model['tf_activation'])(p1)
    c4 = Conv3D(filters=tao_params_model['filters'][3], kernel_size=tao_params_model['kernel_size'],
                strides=tao_params_model['stride'], padding=tao_params_model['padding'],
                activation=tao_params_model['tf_activation'])(c3)
    flat = Flatten()(c4)
    h = Dense(128, activation=tao_params_model['tf_activation'])(flat)
    out = Dense(num_classes, activation='softmax')(h)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=tao_params_model['optimizer'], loss=tao_params_model['loss'],
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy')])
    return model


def get_tan_3d_cnn_model(input_shape=INPUT_SHAPE, num_classes=15, dtype=np.complex64,
                         tensorflow: bool = False, name="tan_3d_cnn"):
    if not tensorflow:
        return _get_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                          name=name)
    else:
        return _get_tf_model(input_shape=input_shape, num_classes=num_classes, name=name)


if __name__ == '__main__':
    model = get_tan_3d_cnn_model(input_shape=INPUT_SHAPE)
    model.summary()
