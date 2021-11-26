import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import concatenate, Add, Activation, Input
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import Recall, Precision, CategoricalAccuracy
from typing import Dict, Optional
from cvnn.losses import ComplexAverageCrossEntropy, ComplexWeightedAverageCrossEntropy
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy, ComplexPrecision, ComplexRecall
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, \
    ComplexMaxPooling2DWithArgmax, ComplexUnPooling2D, ComplexInput, ComplexBatchNormalization, ComplexDense, \
    ComplexUpSampling2D, ComplexConv2DTranspose
from cvnn.activations import cart_softmax, cart_relu
from cvnn.initializers import ComplexHeNormal

IMG_HEIGHT = None  # 128
IMG_WIDTH = None  # 128

DROPOUT_DEFAULT = {  # TODO: Not found yet where
    "downsampling": None,
    "bottle_neck": None,
    "upsampling": None
}

hyper_params = {
    'padding': 'same',
    'kernel_shape': (5, 5),
    'block6_kernel_shape': (1, 1),
    'max_pool_kernel': (2, 2),
    'upsampling_layer': ComplexUnPooling2D,
    'stride': 2,
    'activation': cart_relu,
    'kernels': [12, 24, 48, 96, 192],
    'output_function': cart_softmax,
    'init': ComplexHeNormal(),
    'optimizer': Adam(learning_rate=0.001, beta_1=0.9)
}

tf_hyper_params = {
    'padding': 'same',
    'kernel_shape': (5, 5),
    'block6_kernel_shape': (1, 1),
    'max_pool_kernel': (2, 2),
    'upsampling_layer': UpSampling2D,
    'stride': 2,
    'activation': "relu",
    'kernels': [12, 24, 48, 96, 192],
    'output_function': "softmax",
    'init': HeNormal(),
    'optimizer': Adam(learning_rate=0.001, beta_1=0.9)
}


def _get_downsampling_block(input_to_block, num: int, dtype=np.complex64, dropout=False):
    conv = ComplexConv2D(hyper_params['kernels'][num], hyper_params['kernel_shape'],
                         activation='linear', padding=hyper_params['padding'],
                         kernel_initializer=hyper_params['init'], dtype=dtype)(input_to_block)
    conv = ComplexBatchNormalization(dtype=dtype)(conv)
    conv = Activation(hyper_params['activation'])(conv)
    pool, pool_argmax = ComplexMaxPooling2DWithArgmax(hyper_params['max_pool_kernel'],
                                                      strides=hyper_params['stride'])(conv)
    if dropout:
        pool = ComplexDropout(rate=dropout, dtype=dtype)(pool)
    return pool, pool_argmax


def _tf_get_downsampling_block(input_to_block, num: int, dropout=False):
    conv = Conv2D(tf_hyper_params['kernels'][num], tf_hyper_params['kernel_shape'], activation=None,
                  padding=tf_hyper_params['padding'], kernel_initializer=tf_hyper_params['init'])(input_to_block)
    conv = BatchNormalization()(conv)
    conv = Activation(tf_hyper_params['activation'])(conv)
    pool = MaxPooling2D(tf_hyper_params['max_pool_kernel'], strides=tf_hyper_params['stride'])(conv)
    if dropout:
        pool = Dropout(rate=dropout)(pool)
    return pool


def _get_upsampling_block(input_to_block, pool_argmax, kernels, num: int,
                          upsampling_layer=hyper_params['upsampling_layer'],
                          activation=hyper_params['activation'], dropout=False, dtype=np.complex64):
    if isinstance(upsampling_layer, ComplexUnPooling2D) or upsampling_layer == ComplexUnPooling2D:
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
    elif isinstance(upsampling_layer, ComplexUpSampling2D) or upsampling_layer == ComplexUnPooling2D:
        unpool = ComplexUpSampling2D(size=2)(input_to_block)
    elif isinstance(upsampling_layer, ComplexConv2DTranspose) or upsampling_layer == ComplexConv2DTranspose:
        unpool = ComplexConv2DTranspose(filters=hyper_params["kernels"][num], kernel_size=3)(input_to_block)
    else:
        raise ValueError(f"Upsampling method {upsampling_layer.name} not supported")
    conv = ComplexConv2D(kernels, hyper_params['kernel_shape'],
                         activation='linear', padding=hyper_params['padding'],
                         kernel_initializer=hyper_params['init'], dtype=dtype)(unpool)
    conv = ComplexBatchNormalization(dtype=dtype)(conv)
    conv = Activation(activation)(conv)
    if dropout:
        conv = ComplexDropout(rate=dropout, dtype=dtype)(conv)
    return conv


def _get_tf_upsampling_block(input_to_block, kernels, num: int, upsampling_layer=tf_hyper_params['upsampling_layer'],
                             activation=tf_hyper_params['activation'], dropout=False):
    if isinstance(upsampling_layer, UpSampling2D) or UpSampling2D == upsampling_layer:
        unpool = UpSampling2D(size=2)(input_to_block)
    elif isinstance(upsampling_layer, Conv2DTranspose) or Conv2DTranspose == upsampling_layer:
        # import pdb; pdb.set_trace()
        unpool = Conv2DTranspose(filters=hyper_params["kernels"][num], kernel_size=3)(input_to_block)
    else:
        # import pdb; pdb.set_trace()
        raise ValueError(f"Upsampling method {upsampling_layer.name} not supported")
    conv = Conv2D(kernels, tf_hyper_params['kernel_shape'], activation=None, padding=tf_hyper_params['padding'],
                  kernel_initializer=tf_hyper_params['init'])(unpool)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    if dropout:
        conv = Dropout(rate=dropout)(conv)
    return conv


def _get_my_model(in1, get_downsampling_block, get_upsampling_block, dtype=np.complex64, name="my_own_model",
                  dropout_dict=None, num_classes=4, weights=None):
    # Downsampling
    if dropout_dict is None:
        dropout_dict = DROPOUT_DEFAULT
    pool1, pool1_argmax = get_downsampling_block(in1, 0, dtype=dtype, dropout=dropout_dict["downsampling"])  # Block 1
    pool2, pool2_argmax = get_downsampling_block(pool1, 1, dtype=dtype, dropout=dropout_dict["downsampling"])  # Block 2
    pool3, pool3_argmax = get_downsampling_block(pool2, 2, dtype=dtype, dropout=dropout_dict["downsampling"])  # Block 3
    pool4, pool4_argmax = get_downsampling_block(pool3, 3, dtype=dtype, dropout=dropout_dict["downsampling"])  # Block 4
    pool5, pool5_argmax = get_downsampling_block(pool4, 4, dtype=dtype, dropout=dropout_dict["downsampling"])  # Block 5

    # Bottleneck
    # Block 6
    conv6 = ComplexConv2D(hyper_params['kernels'][4], (1, 1),
                          activation=hyper_params['activation'], padding=hyper_params['padding'],
                          dtype=dtype)(pool5)
    if dropout_dict["bottle_neck"] is not None:
        conv6 = ComplexDropout(rate=dropout_dict["bottle_neck"], dtype=dtype)(conv6)

    # Upsampling
    # Block7
    conv7 = get_upsampling_block(conv6, pool5_argmax, hyper_params['kernels'][3], num=4,
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, pool4_argmax, hyper_params['kernels'][2], num=3,
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, pool3_argmax, hyper_params['kernels'][1], num=2,
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, pool2_argmax, hyper_params['kernels'][0], num=1,
                                  dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 11
    add11 = Add()([conv10, pool1])
    out = get_upsampling_block(add11, pool1_argmax, dropout=False, num=0,
                               kernels=num_classes, activation=hyper_params['output_function'],
                               dtype=dtype)

    if weights is not None:
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = ComplexAverageCrossEntropy()

    model = Model(inputs=[in1], outputs=[out], name=name)
    model.compile(optimizer=hyper_params['optimizer'], loss=loss,
                  metrics=[ComplexCategoricalAccuracy(name='accuracy'),
                           ComplexAverageAccuracy(name='average_accuracy'),
                           ComplexPrecision(name='precision'),
                           ComplexRecall(name='recall')
                           ])
    return model


def _get_my_model_with_tf(in1, get_downsampling_block=_tf_get_downsampling_block,
                          get_upsampling_block=_get_tf_upsampling_block, name="my_own_model",
                          dropout_dict=None, num_classes=4, weights=None):
    # Downsampling
    if dropout_dict is None:
        dropout_dict = DROPOUT_DEFAULT
    pool1 = get_downsampling_block(in1, 0, dropout=dropout_dict["downsampling"])  # Block 1
    pool2 = get_downsampling_block(pool1, 1, dropout=dropout_dict["downsampling"])  # Block 2
    pool3 = get_downsampling_block(pool2, 2, dropout=dropout_dict["downsampling"])  # Block 3
    pool4 = get_downsampling_block(pool3, 3, dropout=dropout_dict["downsampling"])  # Block 4
    pool5 = get_downsampling_block(pool4, 4, dropout=dropout_dict["downsampling"])  # Block 5

    # Bottleneck
    # Block 6
    conv6 = Conv2D(hyper_params['kernels'][4], (1, 1),
                   activation=tf_hyper_params['activation'], padding=tf_hyper_params['padding'])(pool5)
    if dropout_dict["bottle_neck"] is not None:
        conv6 = Dropout(rate=dropout_dict["bottle_neck"])(conv6)

    # Upsampling
    # Block7
    conv7 = get_upsampling_block(conv6, tf_hyper_params['kernels'][3], dropout=dropout_dict["upsampling"], num=4)
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, tf_hyper_params['kernels'][2], dropout=dropout_dict["upsampling"], num=3)
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, tf_hyper_params['kernels'][1], dropout=dropout_dict["upsampling"], num=2)
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, tf_hyper_params['kernels'][0], dropout=dropout_dict["upsampling"], num=1)
    # Block 11
    add11 = Add()([conv10, pool1])
    out = get_upsampling_block(add11, dropout=False, kernels=num_classes, activation=tf_hyper_params['output_function'],
                               num=0)

    if weights is not None:
        print("WARNING: loss function will not be from tensorflow")
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = CategoricalCrossentropy()

    model = Model(inputs=[in1], outputs=[out], name=name)
    model.compile(optimizer=tf_hyper_params['optimizer'], loss=loss,
                  metrics=[
                      CategoricalAccuracy(name='accuracy'),
                      ComplexCategoricalAccuracy(name='complex_accuracy'),
                      ComplexAverageAccuracy(name='average_accuracy'),
                      Precision(name='precision'),
                      Recall(name='recall')
                  ])
    return model


def get_my_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=4, dtype=np.complex64,
                      tensorflow: bool = False,
                      name="my_model", dropout_dict=None, weights=None, hyper_dict: Optional[Dict[str, str]] = None):
    if hyper_dict is not None:
        for key, value in hyper_params:
            if key in hyper_params:
                hyper_params[key] = value
            else:
                print(f"WARGNING: parameter {key} is not used")
    if dropout_dict is None:
        dropout_dict = DROPOUT_DEFAULT
    if not tensorflow:
        in1 = complex_input(shape=input_shape, dtype=dtype)
        return _get_my_model(in1, _get_downsampling_block, _get_upsampling_block, dtype=dtype, name=name,
                             dropout_dict=dropout_dict, num_classes=num_classes, weights=weights)
    else:
        in1 = Input(shape=input_shape)
        return _get_my_model_with_tf(in1, _tf_get_downsampling_block, _get_tf_upsampling_block, name="tf_" + name,
                                     dropout_dict=dropout_dict, num_classes=num_classes, weights=weights)
