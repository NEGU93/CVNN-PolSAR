import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, Add, Activation
from tensorflow.keras import Model, Sequential
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
    'kernel_shape': (3, 3),
    'block6_kernel_shape': (1, 1),
    'max_pool_kernel': (2, 2),
    'upsampling_layer': ComplexUnPooling2D,
    'stride': 2,
    'activation': cart_relu,
    'kernels': [12, 24, 48, 96, 192],
    'output_function': cart_softmax,
    'init': ComplexHeNormal(),
    'optimizer': Adam(learning_rate=0.0001, beta_1=0.9)
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


def _get_upsampling_block(input_to_block, pool_argmax, kernels, upsampling_layer=hyper_params['upsampling_layer'],
                          activation=hyper_params['activation'], dropout=False, dtype=np.complex64):
    if isinstance(upsampling_layer, ComplexUnPooling2D):
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
    elif isinstance(upsampling_layer, ComplexUpSampling2D):
        unpool = ComplexUpSampling2D(size=2)(input_to_block)
    elif isinstance(upsampling_layer, ComplexConv2DTranspose):
        unpool = ComplexConv2DTranspose(filters=tf.shape(input_to_block)[-1], kernel_size=3)
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


def _get_my_model(in1, get_downsampling_block, get_upsampling_block, dtype=np.complex64, name="cao_model",
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
    if dropout_dict["bottle_neck"]:
        conv6 = ComplexDropout(rate=dropout_dict["bottle_neck"], dtype=dtype)(conv6)

    # Upsampling
    # Block7
    conv7 = get_upsampling_block(conv6, pool5_argmax, hyper_params['kernels'][3],
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, pool4_argmax, hyper_params['kernels'][2],
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, pool3_argmax, hyper_params['kernels'][1],
                                 dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, pool2_argmax, hyper_params['kernels'][0],
                                  dropout=dropout_dict["upsampling"], dtype=dtype)
    # Block 11
    add11 = Add()([conv10, pool1])
    out = get_upsampling_block(add11, pool1_argmax, dropout=False,
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


def get_my_unet_model(index: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=4, dtype=np.complex64,
                      name="my_model", dropout_dict=None, weights=None):
    if index == 1:
        hyper_params['kernel_shape'] = (5, 5)
    elif index == 2:
        hyper_params['kernel_shape'] = (5, 5)
        hyper_params['upsampling_layer'] = ComplexUpSampling2D
    elif index == 3:
        hyper_params['kernel_shape'] = (5, 5)
        hyper_params['upsampling_layer'] = ComplexConv2DTranspose
    if dropout_dict is None:
        dropout_dict = DROPOUT_DEFAULT
    in1 = complex_input(shape=input_shape, dtype=dtype)
    return _get_my_model(in1, _get_downsampling_block, _get_upsampling_block, dtype=dtype, name=name,
                         dropout_dict=dropout_dict, num_classes=num_classes, weights=weights)
