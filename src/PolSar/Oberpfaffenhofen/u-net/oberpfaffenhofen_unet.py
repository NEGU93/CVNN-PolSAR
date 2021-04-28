import numpy as np
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, MaxPooling2D, UpSampling2D
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, ComplexMaxPooling2D, ComplexUpSampling2D, \
    ComplexMaxPooling2DWithArgmax, ComplexUnPooling2D, ComplexInput
from cvnn.activations import softmax_real_with_avg

IMG_HEIGHT = 128
IMG_WIDTH = 128


def get_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    dtype = np.complex64
    in1 = complex_input(shape=input_shape, dtype=dtype)
    # in1 = ComplexInput(input_shape=input_shape, dtype=dtype)

    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(in1)
    conv1 = ComplexDropout(0.5)(conv1)
    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv1)
    pool1 = ComplexMaxPooling2D((2, 2))(conv1)

    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool1)
    conv2 = ComplexDropout(0.5)(conv2)
    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv2)
    pool2 = ComplexMaxPooling2D((2, 2))(conv2)

    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool2)
    conv3 = ComplexDropout(0.5)(conv3)
    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv3)
    pool3 = ComplexMaxPooling2D((2, 2))(conv3)

    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool3)
    conv4 = ComplexDropout(0.5)(conv4)
    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv4)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv4)
    up1 = concatenate([tmp_up, conv3], axis=-1)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up1)
    conv5 = ComplexDropout(0.5)(conv5)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv5)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv5)
    up2 = concatenate([tmp_up, conv2], axis=-1)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv6 = ComplexDropout(0.5)(conv6)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv6)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv6)
    up2 = concatenate([tmp_up, conv1], axis=-1)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv7 = ComplexDropout(0.5)(conv7)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv7)
    segmentation = ComplexConv2D(3, (1, 1), activation='softmax_real_with_avg', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])
    losses = {'seg': 'categorical_crossentropy'}
    metrics = {'seg': ['acc']}
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    # https://github.com/tensorflow/tensorflow/issues/38988
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


def get_cao_cvfcn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    dtype = np.complex64
    cao_params = {
        'padding': 'same',
        'activation': 'cart_relu'
    }
    in1 = complex_input(shape=input_shape)

    # Downsampling
    conv1 = ComplexConv2D(12, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(in1)
    conv1 = ComplexDropout(0.5)(conv1)
    pool1, pool1_argmax = ComplexMaxPooling2DWithArgmax((2, 2))(conv1)

    conv2 = ComplexConv2D(24, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(pool1)
    conv2 = ComplexDropout(0.5)(conv2)
    pool2, pool2_argmax = ComplexMaxPooling2DWithArgmax((2, 2))(conv2)

    conv3 = ComplexConv2D(48, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(pool2)
    conv3 = ComplexDropout(0.5)(conv3)
    pool3, pool3_argmax = ComplexMaxPooling2DWithArgmax((2, 2))(conv3)

    conv4 = ComplexConv2D(96, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(pool3)
    conv4 = ComplexDropout(0.5)(conv4)
    pool4, pool4_argmax = ComplexMaxPooling2DWithArgmax((2, 2))(conv4)

    conv5 = ComplexConv2D(192, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(pool4)
    conv5 = ComplexDropout(0.5)(conv5)
    pool5, pool5_argmax = ComplexMaxPooling2DWithArgmax((2, 2))(conv5)

    # Bottleneck
    conv = ComplexConv2D(192, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                         dtype=dtype)(pool5)

    # Upsampling
    unpool7 = ComplexUnPooling2D(conv5.shape[1:])([conv, pool5_argmax])
    conv7 = ComplexConv2D(96, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(unpool7)

    add8 = Add()([conv7, pool4])
    unpool8 = ComplexUnPooling2D(conv4.shape[1:])([add8, pool4_argmax])
    conv8 = ComplexConv2D(48, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(unpool8)

    add9 = Add()([conv8, pool3])
    unpool9 = ComplexUnPooling2D(conv3.shape[1:])([add9, pool3_argmax])
    conv9 = ComplexConv2D(24, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                          dtype=dtype)(unpool9)

    add10 = Add()([conv9, pool2])
    unpool10 = ComplexUnPooling2D(conv2.shape[1:])([add10, pool2_argmax])
    conv10 = ComplexConv2D(12, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                           dtype=dtype)(unpool10)

    add11 = Add()([conv10, pool1])
    unpool11 = ComplexUnPooling2D(conv1.shape[1:])([add11, pool1_argmax])
    conv11 = ComplexConv2D(3, (3, 3), activation=cao_params['activation'], padding=cao_params['padding'],
                           dtype=dtype)(unpool11)

    out = softmax_real_with_avg(conv11)

    model = Model(inputs=[in1], outputs=[out])
    losses = {'seg': 'categorical_crossentropy'}
    metrics = {'seg': ['acc']}
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    # https://github.com/tensorflow/tensorflow/issues/38988
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model

if __name__ == '__main__':
    model = get_model()
    plot_model(model, show_shapes=True)
