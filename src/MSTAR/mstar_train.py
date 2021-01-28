from mstar_data_processing import get_train_and_test
from pdb import set_trace
import numpy as np
import tensorflow as tf
import cvnn.layers as complex_layers


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


if __name__ == '__main__':
    img_train, img_test, labels_train, labels_test = get_train_and_test()
    model = get_model()
    history = model.fit(x=img_train.reshape(img_train.shape + (1,)), y=labels_train,
                        validation_data=(img_test.reshape(img_test.shape + (1,)), labels_test),
                        epochs=10)
