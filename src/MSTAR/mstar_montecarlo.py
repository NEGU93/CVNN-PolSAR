from pdb import set_trace
import numpy as np
from cvnn.montecarlo import RealVsComplex
import tensorflow as tf
from cvnn.montecarlo import _save_montecarlo_log
import cvnn.layers as complex_layers
from mstar_data_processing import get_train_and_test

montecarlo_config = {
    'polar': True,
    'iterations': 2,
    'epochs': 2,
    'batch_size': 100
}


def get_own_model():
    input_shape = (128, 128)
    model = tf.keras.models.Sequential(name='complex_model')
    model.add(complex_layers.ComplexInput(input_shape=input_shape + (1,)))  # Always use ComplexInput at the start
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(10, activation='softmax_real'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def do_monte_carlo(complex_model):
    montecarlo = RealVsComplex(complex_model)
    img_train, img_test, labels_train, labels_test = get_train_and_test()
    img_train = np.array(img_train.reshape(img_train.shape + (1,)))
    img_test = np.array(img_test.reshape(img_test.shape + (1,)))
    montecarlo.run(x=img_train, y=labels_train, validation_data=(img_test, labels_test), polar=montecarlo_config['polar'],
                   iterations=montecarlo_config['iterations'], epochs=montecarlo_config['epochs'],
                   batch_size=montecarlo_config['batch_size'],
                   data_summary="MSTAR 10 class database",
                   debug=False)


if __name__ == '__main__':
    complex_model = get_own_model()
    do_monte_carlo(complex_model)
