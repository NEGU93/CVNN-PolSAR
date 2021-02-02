from mstar_data_processing import get_train_and_test
from pdb import set_trace
import numpy as np
import tensorflow as tf
import cvnn.layers as complex_layers
import tensorflow.keras.layers as layers


def get_model(verbose: bool = False):
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(128, 128, 1)))  # Always use ComplexInput at the start
    model.add(complex_layers.ComplexConv2D(6, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(12, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(108, activation='cart_relu'))
    model.add(complex_layers.ComplexDense(10, activation='softmax_real'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def get_tf_ding_model(verbose: bool = False, input_shape=(128, 128)):
    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=input_shape + (1,)))  # Always use ComplexInput at the start

    model.add(layers.Conv2D(96, (3, 3), activation='relu'))
    model.add(layers.Conv2D(96, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=1))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=1))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def get_ding_model(verbose: bool = False, input_shape=(128, 128), dtype=np.complex64):
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=input_shape + (1,),
                                          dtype=dtype))  # Always use ComplexInput at the start

    model.add(complex_layers.ComplexConv2D(96, (3, 3), activation='cart_relu', dtype=dtype))
    model.add(complex_layers.ComplexConv2D(96, (3, 3), activation='cart_relu', dtype=dtype))

    model.add(complex_layers.ComplexMaxPooling2D((2, 2), strides=1, dtype=dtype))

    model.add(complex_layers.ComplexConv2D(256, (3, 3), activation='cart_relu', dtype=dtype))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), strides=1, dtype=dtype))

    model.add(complex_layers.ComplexFlatten(dtype=dtype))
    model.add(complex_layers.ComplexDense(10, activation='softmax_real', dtype=dtype))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def get_chen_model(verbose: bool = False, input_shape=(128, 128), dtype=np.complex64):
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=input_shape + (1,), dtype=dtype))  # Always use ComplexInput at the start

    model.add(complex_layers.ComplexConv2D(16, (5, 5), activation='cart_relu', dtype=dtype))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), strides=2, dtype=dtype))

    model.add(complex_layers.ComplexConv2D(32, (5, 5), activation='cart_relu', dtype=dtype))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), strides=2, dtype=dtype))

    model.add(complex_layers.ComplexConv2D(64, (6, 6), activation='cart_relu', dtype=dtype))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), strides=2, dtype=dtype))

    model.add(complex_layers.ComplexDropout(0.5, dtype=dtype))
    model.add(complex_layers.ComplexConv2D(128, (5, 5), activation='cart_relu', dtype=dtype))

    model.add(complex_layers.ComplexConv2D(10, (3, 3), activation='softmax_real', dtype=dtype))
    # model.add(complex_layers.ComplexFlatten())
    # model.add(complex_layers.ComplexDense(10, activation='softmax_real'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


if __name__ == '__main__':
    model = 'ding'
    dtype = np.float32
    if model == 'chen':
        input_shape = (88, 88)
        model = get_chen_model(input_shape=input_shape, verbose=True, dtype=dtype)
    elif model == 'ding':
        input_shape = (128, 128)
        # model = get_tf_ding_model(input_shape=input_shape, verbose=True)
        model = get_ding_model(input_shape=input_shape, verbose=True, dtype=dtype)
    else:
        input_shape = (128, 128)
        model = get_model(True)
    img_train, img_test, labels_train, labels_test = get_train_and_test(input_shape=input_shape)
    img_train = np.array(img_train.reshape(img_train.shape + (1,)))
    img_test = np.array(img_test.reshape(img_test.shape + (1,)))
    if dtype == np.float32:
        img_train == np.abs(img_train)
        img_test == np.abs(img_test)
    history = model.fit(x=img_train, y=labels_train,
                        validation_data=(img_test, labels_test),
                        epochs=10)
