import numpy as np
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, MaxPooling2D, UpSampling2D
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, ComplexMaxPooling2D, ComplexUpSampling2D

IMG_HEIGHT = 128
IMG_WIDTH = 128


def get_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    dtype = np.complex64
    in1 = complex_input(shape=input_shape, dtype='complex64')

    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(in1)
    conv1 = ComplexDropout(0.2)(conv1)
    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv1)
    pool1 = ComplexMaxPooling2D((2, 2))(conv1)

    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool1)
    conv2 = ComplexDropout(0.2)(conv2)
    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv2)
    pool2 = ComplexMaxPooling2D((2, 2))(conv2)

    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool2)
    conv3 = ComplexDropout(0.2)(conv3)
    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv3)
    pool3 = ComplexMaxPooling2D((2, 2))(conv3)

    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool3)
    conv4 = ComplexDropout(0.2)(conv4)
    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv4)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv4)
    up1 = concatenate([tmp_up, conv3], axis=-1)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up1)
    conv5 = ComplexDropout(0.2)(conv5)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv5)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv5)
    up2 = concatenate([tmp_up, conv2], axis=-1)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv6 = ComplexDropout(0.2)(conv6)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv6)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv6)
    up2 = concatenate([tmp_up, conv1], axis=-1)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv7 = ComplexDropout(0.2)(conv7)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv7)
    segmentation = ComplexConv2D(3, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])
    losses = {'seg': 'binary_crossentropy'}
    metrics = {'seg': ['acc']}
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    # https://github.com/tensorflow/tensorflow/issues/38988
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


if __name__ == '__main__':
    model = get_model()
    plot_model(model, show_shapes=True)
