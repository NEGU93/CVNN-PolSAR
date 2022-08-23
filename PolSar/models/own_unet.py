import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Concatenate, Add, Activation, Input
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, BatchNormalization, MaxPooling2D, \
    UpSampling2D, AvgPool2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import Recall, Precision, CategoricalAccuracy
from typing import Dict, Optional, Any
from cvnn.losses import ComplexAverageCrossEntropy, ComplexWeightedAverageCrossEntropy
from cvnn.metrics import ComplexCategoricalAccuracy, ComplexAverageAccuracy, ComplexPrecision, ComplexRecall
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, \
    ComplexMaxPooling2DWithArgmax, ComplexUnPooling2D, ComplexInput, ComplexBatchNormalization, ComplexDense, \
    ComplexUpSampling2D, ComplexConv2DTranspose, ComplexAvgPooling2D, ComplexPolarAvgPooling2D, ComplexMaxPooling2D
from cvnn.activations import cart_softmax, cart_relu
from cvnn.initializers import ComplexHeNormal

IMG_HEIGHT = None  # 128
IMG_WIDTH = None  # 128

DROPOUT_DEFAULT = {
    "downsampling": None,
    "bottle_neck": None,
    "upsampling": None
}

hyper_params = {
    'padding': 'same',
    'consecutive_conv_layers': 0,
    'kernel_shape': (3, 3),
    'block6_kernel_shape': (1, 1),
    'max_pool_kernel': (2, 2),
    'concat': Add,
    'upsampling_layer': ComplexUnPooling2D,
    'stride': 2,
    'pooling': ComplexMaxPooling2DWithArgmax,
    'activation': cart_relu,
    'kernels': [12, 24, 48, 96, 192],
    'output_function': cart_softmax,
    'init': ComplexHeNormal,
    'optimizer': Adam,
    'learning_rate': 0.0001,
    'depth': 5
}

complex_cast_to_tf = {
    'upsampling_layer': {
        ComplexUpSampling2D: UpSampling2D,
        ComplexUnPooling2D: ComplexUnPooling2D,
        ComplexConv2DTranspose: Conv2DTranspose
    },
    'pooling': {
        ComplexMaxPooling2DWithArgmax: ComplexMaxPooling2DWithArgmax,
        ComplexAvgPooling2D: AvgPool2D,
        ComplexMaxPooling2D: MaxPooling2D
    },
    'init': {ComplexHeNormal: HeNormal}     # TODO: Not activation cast done yet.
}


def _get_downsampling_block(input_to_block, num: int, dtype=np.complex64, dropout: Optional[bool] = False):
    conv = ComplexConv2D(hyper_params['kernels'][:hyper_params['depth']][num], hyper_params['kernel_shape'],
                         activation='linear', padding=hyper_params['padding'],
                         kernel_initializer=hyper_params['init'](), dtype=dtype)(input_to_block)
    for _ in range(hyper_params['consecutive_conv_layers']):
        conv = ComplexConv2D(hyper_params['kernels'][:hyper_params['depth']][num], hyper_params['kernel_shape'],
                             activation='linear', padding=hyper_params['padding'],
                             kernel_initializer=hyper_params['init'](), dtype=dtype)(conv)
    conv = ComplexBatchNormalization(dtype=dtype)(conv)
    conv = Activation(hyper_params['activation'])(conv)
    if hyper_params['pooling'] == ComplexMaxPooling2DWithArgmax:
        pool, pool_argmax = ComplexMaxPooling2DWithArgmax(hyper_params['max_pool_kernel'],
                                                          strides=hyper_params['stride'])(conv)
    elif hyper_params['pooling'] == ComplexAvgPooling2D:
        pool = ComplexAvgPooling2D(hyper_params['max_pool_kernel'], strides=hyper_params['stride'])(conv)
        pool_argmax = None
    elif hyper_params['pooling'] == ComplexPolarAvgPooling2D:
        pool = ComplexPolarAvgPooling2D(hyper_params['max_pool_kernel'], strides=hyper_params['stride'])(conv)
        pool_argmax = None
    else:
        raise ValueError(f"Unknown pooling {hyper_params['pooling']}")
    if dropout:
        pool = ComplexDropout(rate=dropout, dtype=dtype)(pool)
    return pool, pool_argmax


def _tf_get_downsampling_block(input_to_block, num: int, dropout: Optional[bool] = False):
    conv = Conv2D(hyper_params['kernels'][:hyper_params['depth']][num], hyper_params['kernel_shape'], activation=None,
                  padding=hyper_params['padding'],
                  kernel_initializer=complex_cast_to_tf['init'][hyper_params['init']]())(input_to_block)
    for _ in range(hyper_params['consecutive_conv_layers']):
        conv = Conv2D(hyper_params['kernels'][:hyper_params['depth']][num], hyper_params['kernel_shape'], activation=None,
                      padding=hyper_params['padding'],
                      kernel_initializer=complex_cast_to_tf['init'][hyper_params['init']]())(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(hyper_params['activation'])(conv)
    if complex_cast_to_tf['pooling'][hyper_params['pooling']] == ComplexMaxPooling2DWithArgmax:
        pool, pool_argmax = ComplexMaxPooling2DWithArgmax(hyper_params['max_pool_kernel'],
                                                          strides=hyper_params['stride'])(conv)
    elif complex_cast_to_tf['pooling'][hyper_params['pooling']] == AvgPool2D:
        pool = AvgPool2D(hyper_params['max_pool_kernel'], strides=hyper_params['stride'])(conv)
        pool_argmax = None
    elif complex_cast_to_tf['pooling'][hyper_params['pooling']] == MaxPooling2D:
        pool = MaxPooling2D(hyper_params['max_pool_kernel'], strides=hyper_params['stride'])(conv)
        pool_argmax = None
    else:
        raise ValueError(f"Unknown pooling {hyper_params['pooling']}")
    if dropout:
        pool = Dropout(rate=dropout)(pool)
    return pool, pool_argmax


def _get_upsampling_block(input_to_block, pool_argmax, kernels, num: int, activation,
                          dropout: Optional[bool] = False, dtype=np.complex64):
    if hyper_params['upsampling_layer'] == ComplexUnPooling2D:
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
    elif hyper_params['upsampling_layer'] == ComplexUpSampling2D:
        unpool = ComplexUpSampling2D(size=2)(input_to_block)
    elif hyper_params['upsampling_layer'] == ComplexConv2DTranspose:
        unpool = ComplexConv2DTranspose(filters=num, kernel_size=3, strides=(2, 2), padding='same',
                                        dilation_rate=(1, 1))(input_to_block)
    else:
        raise ValueError(f"Upsampling method {hyper_params['upsampling_layer'].name} not supported")
    conv = ComplexConv2D(kernels, hyper_params['kernel_shape'],
                         activation='linear', padding=hyper_params['padding'],
                         kernel_initializer=hyper_params['init'](), dtype=dtype)(unpool)
    for _ in range(hyper_params['consecutive_conv_layers']):
        conv = ComplexConv2D(kernels, hyper_params['kernel_shape'],
                             activation='linear', padding=hyper_params['padding'],
                             kernel_initializer=hyper_params['init'](), dtype=dtype)(conv)
    conv = ComplexBatchNormalization(dtype=dtype)(conv)
    conv = Activation(activation)(conv)
    if dropout:
        conv = ComplexDropout(rate=dropout, dtype=dtype)(conv)
    return conv


def _get_tf_upsampling_block(input_to_block, pool_argmax, kernels, num: int,
                             activation, dropout: Optional[bool] = False):
    if UpSampling2D == complex_cast_to_tf['upsampling_layer'][hyper_params['upsampling_layer']]:
        unpool = UpSampling2D(size=2)(input_to_block)
    elif Conv2DTranspose == complex_cast_to_tf['upsampling_layer'][hyper_params['upsampling_layer']]:
        unpool = Conv2DTranspose(filters=num, kernel_size=3, strides=(2, 2), padding='same',
                                 dilation_rate=(1, 1))(input_to_block)
    elif complex_cast_to_tf['upsampling_layer'][hyper_params['upsampling_layer']] == ComplexUnPooling2D:
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
    else:
        raise ValueError(f"Upsampling method"
                         f" {complex_cast_to_tf['upsampling_layer'][hyper_params['upsampling_layer']].name} "
                         f"not supported")
    conv = Conv2D(kernels, hyper_params['kernel_shape'], activation=None, padding=hyper_params['padding'],
                  kernel_initializer=complex_cast_to_tf['init'][hyper_params['init']]())(unpool)
    for _ in range(hyper_params['consecutive_conv_layers']):
        conv = Conv2D(kernels, hyper_params['kernel_shape'], activation=None, padding=hyper_params['padding'],
                      kernel_initializer=complex_cast_to_tf['init'][hyper_params['init']]())(conv)
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

    pool = in1
    pools = []
    argmax_pools = []
    for index in range(len(hyper_params['kernels'][:hyper_params['depth']])):
        pool, pool_argmax = get_downsampling_block(pool, index, dtype=dtype, dropout=dropout_dict["downsampling"])
        pools.append(pool)
        argmax_pools.append(pool_argmax)

    # Bottleneck
    index = -1
    conv = ComplexConv2D(hyper_params['kernels'][:hyper_params['depth']][index], (1, 1),
                         activation=hyper_params['activation'], padding=hyper_params['padding'],
                         dtype=dtype)(pools.pop())
    if dropout_dict["bottle_neck"] is not None:
        conv = ComplexDropout(rate=dropout_dict["bottle_neck"], dtype=dtype)(conv)

    # Upsampling
    while pools:
        index -= 1
        pool = pools.pop()
        pool_argmax = argmax_pools.pop()
        conv = get_upsampling_block(conv, pool_argmax, hyper_params['kernels'][:hyper_params['depth']][index], num=4,
                                    activation=hyper_params['activation'],
                                    dropout=dropout_dict["upsampling"], dtype=dtype)
        if hyper_params['concat'] == Concatenate:
            conv = Concatenate()([conv, pool])
        elif hyper_params['concat'] == Add:
            conv = Add()([conv, pool])
        else:
            raise KeyError(f"Concatenation {hyper_params['concat']} not known")
    out = get_upsampling_block(conv, argmax_pools.pop(), activation=hyper_params['output_function'], dropout=False,
                               num=0, kernels=num_classes, dtype=dtype)

    if weights is not None:
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = ComplexAverageCrossEntropy()

    model = Model(inputs=[in1], outputs=[out], name=name)
    model.compile(optimizer=hyper_params['optimizer'](learning_rate=hyper_params['learning_rate']), loss=loss,
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

    pool = in1
    pools = []
    argmax_pools = []
    for index in range(len(hyper_params['kernels'][:hyper_params['depth']])):
        pool, pool_argmax = get_downsampling_block(pool, index, dropout=dropout_dict["downsampling"])
        pools.append(pool)
        argmax_pools.append(pool_argmax)

    # Bottleneck
    index = -1
    kernel_backup = hyper_params['kernels'][:hyper_params['depth']][index]
    conv = Conv2D(kernel_backup, (1, 1),  activation=hyper_params['activation'],
                  padding=hyper_params['padding'])(pools.pop())
    if dropout_dict["bottle_neck"] is not None:
        conv = Dropout(rate=dropout_dict["bottle_neck"])(conv)

    # Upsampling
    while pools:
        pool = pools.pop()
        pool_argmax = argmax_pools.pop()
        index -= 1
        new_kernel = hyper_params['kernels'][:hyper_params['depth']][index]
        conv = get_upsampling_block(conv, pool_argmax, new_kernel,
                                    num=kernel_backup,
                                    activation=hyper_params['activation'],
                                    dropout=dropout_dict["upsampling"])
        kernel_backup = new_kernel
        if hyper_params['concat'] == Concatenate:
            conv = Concatenate()([conv, pool])
        elif hyper_params['concat'] == Add:
            conv = Add()([conv, pool])
        else:
            raise KeyError(f"Concatenation {hyper_params['concat']} not known")
    out = get_upsampling_block(conv, argmax_pools.pop(), activation=hyper_params['output_function'], dropout=False,
                               num=kernel_backup, kernels=num_classes)

    if weights is not None:
        print("WARNING: loss function will not be from tensorflow")
        loss = ComplexWeightedAverageCrossEntropy(weights=weights)
    else:
        loss = CategoricalCrossentropy()

    model = Model(inputs=[in1], outputs=[out], name=name)
    model.compile(optimizer=hyper_params['optimizer'](learning_rate=hyper_params['learning_rate']), loss=loss,
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
                      name="my_model", dropout_dict=None, weights=None, hyper_dict: Optional[Dict] = None):
    if hyper_dict is not None:
        for key, value in hyper_dict.items():
            if key in hyper_params.keys():
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


def get_my_unet_tests(index: int, depth=5, *args, **kwargs):
    if index == 0 or index is None:
        return get_my_unet_model(*args, hyper_dict={'learning_rate': 0.1, 'depth': depth}, **kwargs)
    elif index == 1:
        return get_my_unet_model(*args, hyper_dict={'depth': depth}, **kwargs)
    elif index == 2:        # Good peak.
        return get_my_unet_model(*args, hyper_dict={'consecutive_conv_layers': 1, 'depth': depth}, **kwargs)
    elif index == 3:
        return get_my_unet_model(*args, hyper_dict={'consecutive_conv_layers': 1,
                                                    'learning_rate': 0.00001, 'depth': depth}, **kwargs)
    elif index == 4:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D, 'depth': depth}, **kwargs)
    elif index == 5:            # Best ending, with apparently needs more epochs
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 6:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexConv2DTranspose, 'depth': depth}, **kwargs)
    elif index == 7:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexConv2DTranspose,
                                                    'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 8:        # Good peak.
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (5, 5), 'depth': depth}, **kwargs)
    elif index == 9:
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (5, 5), 'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 10:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'concat': Concatenate, 'depth': depth},
                                 **kwargs)
    elif index == 11:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'concat': Concatenate, 'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 12:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexConv2DTranspose,
                                                    'concat': Concatenate, 'depth': depth},
                                 **kwargs)
    elif index == 13:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexConv2DTranspose,
                                                    'concat': Concatenate, 'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 14:           # Good one
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'concat': Concatenate,
                                                    'consecutive_conv_layers': 1, 'depth': depth},
                                 **kwargs)
    elif index == 15:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'concat': Concatenate,
                                                    'consecutive_conv_layers': 1,
                                                    'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 16:           # VALIDATION WINNER
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexAvgPooling2D,
                                                    'learning_rate': 0.00001, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 17:           # Good one
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexAvgPooling2D,
                                                    'concat': Concatenate,
                                                    'learning_rate': 0.00001, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 18:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexAvgPooling2D,
                                                    'concat': Concatenate,
                                                    'learning_rate': 0.00001,
                                                    'consecutive_conv_layers': 1, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 19:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexPolarAvgPooling2D,
                                                    'learning_rate': 0.00001, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 20:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexPolarAvgPooling2D,
                                                    'concat': Concatenate,
                                                    'learning_rate': 0.00001, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 21:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexPolarAvgPooling2D,
                                                    'concat': Concatenate,
                                                    'learning_rate': 0.00001,
                                                    'consecutive_conv_layers': 1, 'depth': depth,
                                                    },
                                 **kwargs)
    elif index == 22:  # Good peak.
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (7, 7), 'depth': depth,}, **kwargs)
    elif index == 23:
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (7, 7), 'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 24:  # Good peak.
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (9, 9), 'depth': depth,}, **kwargs)
    elif index == 25:
        return get_my_unet_model(*args, hyper_dict={'kernel_shape': (9, 9), 'learning_rate': 0.00001, 'depth': depth},
                                 **kwargs)
    elif index == 26:        # Good peak.
        return get_my_unet_model(*args, hyper_dict={'consecutive_conv_layers': 2, 'depth': depth,}, **kwargs)
    elif index == 27:
        return get_my_unet_model(*args, hyper_dict={'consecutive_conv_layers': 2,
                                                    'learning_rate': 0.00001, 'depth': depth,}, **kwargs)
    elif index == 28:
        return get_my_unet_model(*args, hyper_dict={'upsampling_layer': ComplexUpSampling2D,
                                                    'pooling': ComplexAvgPooling2D,
                                                    'consecutive_conv_layers': 1,
                                                    'learning_rate': 0.00001, 'depth': depth,
                                                    },
                                 **kwargs)
    else:
        raise ValueError(f"{index} index out of range.")
