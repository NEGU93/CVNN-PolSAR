import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, AveragePooling2D
from cvnn.layers import complex_input, ComplexConv2D, ComplexAvgPooling2D, ComplexFlatten, ComplexDense
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy
from cvnn.losses import ComplexMeanSquareError
from cvnn.activations import cart_softmax


IMG_HEIGHT = 12
IMG_WIDTH = 12

zhang_params_model = {
    'padding': 'valid',
    'kernel_size':  3,
    'stride': 1,
    'complex_filters': [6, 12],
    'real_filters': [8, 22],
    'pool_size': 2,
    'loss': ComplexMeanSquareError(),       # End of II.A.4
    'tf_loss': MeanSquaredError(),
    'activation': 'cart_sigmoid',           # "Note that the sigmoid function is used in this paper." but cart or polar?
    'optimizer': SGD(learning_rate=0.08)    # Start of II.B
}


class CustomAccuracy(ComplexCategoricalAccuracy):

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = cart_softmax(y_pred)
        super(CustomAccuracy, self).update_state(y_true, y_pred,
                                                 sample_weight=sample_weight, ignore_unlabeled=ignore_unlabeled)


class CustomAverageAccuracy(ComplexAverageAccuracy):

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = cart_softmax(y_pred)
        super(CustomAverageAccuracy, self).update_state(y_true, y_pred,
                                                        sample_weight=sample_weight)


def _get_model(input_shape, num_classes, dtype, name='zhang_cnn'):
    if dtype.is_complex:
        filters = "complex_filters"
    else:
        filters = "real_filters"
    in1 = complex_input(shape=input_shape, dtype=dtype)
    c1 = ComplexConv2D(filters=zhang_params_model[filters][0], kernel_size=zhang_params_model['kernel_size'],
                       strides=zhang_params_model['stride'], padding=zhang_params_model['padding'],
                       activation=zhang_params_model['activation'], dtype=dtype)(in1)
    p1 = ComplexAvgPooling2D(pool_size=zhang_params_model['pool_size'], dtype=dtype)(c1)
    c2 = ComplexConv2D(filters=zhang_params_model[filters][1], kernel_size=zhang_params_model['kernel_size'],
                       strides=zhang_params_model['stride'], padding=zhang_params_model['padding'],
                       activation=zhang_params_model['activation'], dtype=dtype)(p1)
    flat = ComplexFlatten(dtype=dtype)(c2)
    out = ComplexDense(num_classes, activation='linear', dtype=dtype)(flat)
    model = Model(inputs=in1, outputs=out, name=name)
    model.compile(optimizer=zhang_params_model['optimizer'], loss=zhang_params_model['loss'],
                  metrics=[CustomAccuracy(name='accuracy'),
                           CustomAverageAccuracy(name='average_accuracy')])
    return model


def _get_tf_model(input_shape, num_classes, dtype, name='tf_zhang_cnn'):
    if dtype.is_complex:
        raise ValueError(f"Cannot use Tensorflow for creating a complex model")
    filters = "real_filters"
    in1 = Input(shape=input_shape)
    c1 = Conv2D(filters=zhang_params_model[filters][0], kernel_size=zhang_params_model['kernel_size'],
                strides=zhang_params_model['stride'], padding=zhang_params_model['padding'],
                activation=zhang_params_model['activation'])(in1)
    p1 = AveragePooling2D(pool_size=zhang_params_model['pool_size'])(c1)
    c2 = Conv2D(filters=zhang_params_model[filters][1], kernel_size=zhang_params_model['kernel_size'],
                strides=zhang_params_model['stride'], padding=zhang_params_model['padding'],
                activation=zhang_params_model['activation'])(p1)
    flat = Flatten(dtype=dtype)(c2)
    out = Dense(num_classes, activation='linear')(flat)
    model = Model(inputs=[in1], outputs=[out], name=name)
    model.compile(optimizer=zhang_params_model['optimizer'], loss=zhang_params_model['tf_loss'],
                  metrics=[CategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy'),
                           ])


def get_zhang_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 6), num_classes=15, dtype=np.complex64,
                        tensorflow: bool = False, name="my_model"):
    if not tensorflow:
        return _get_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                          name=name)
    else:
        return _get_tf_model(input_shape=input_shape, num_classes=num_classes, dtype=tf.dtypes.as_dtype(dtype),
                             name=name)


if __name__ == '__main__':
    model = get_zhang_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 9))
    model.summary()
