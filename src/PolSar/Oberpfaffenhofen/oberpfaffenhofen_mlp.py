from pdb import set_trace
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from typing import Optional
# own modules
from cvnn.utils import randomize
from cvnn.dataset import Dataset
from cvnn.montecarlo import RealVsComplex
from cvnn import layers as complex_layers
from oberpfaffenhofen_dataset import get_dataset_for_classification, separate_train_test_pixels


def run_monte(dataset, validation_data, test_data, iterations=10, epochs=200, batch_size=100,
              optimizer='sgd', shape_raw=None, activation='cart_relu',
              polar=False, dropout: Optional[float] = 0.5):
    if shape_raw is None:
        shape_raw = [50]

    # Create complex network
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output
    if len(shape_raw) == 0:
        print("No hidden layers are used. activation and dropout will be ignored")
        shape = [
            complex_layers.InputLayer(input_shape=input_size, dtype=np.complex64),
            complex_layers.ComplexDense(units=output_size, activation='softmax_real', dtype=np.complex64)
        ]
    else:  # len(shape_raw) > 0:
        shape = [complex_layers.InputLayer(input_shape=input_size, dtype=np.complex64)]
        for i in range(0, len(shape_raw)):
            shape.append(complex_layers.ComplexDense(units=shape_raw[i], activation=activation))
            if dropout is not None:
                shape.append(complex_layers.ComplexDropout(rate=dropout))
        shape.append(complex_layers.ComplexDense(units=output_size, activation='softmax_real'))

    complex_network = Sequential(name="complex_network", layers=shape)
    complex_network.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

    monte_carlo = RealVsComplex(complex_network)
    for model in monte_carlo.models:
        model.summary()
    monte_carlo.output_config['plot_all'] = True
    monte_carlo.output_config['confusion_matrix'] = True
    monte_carlo.run(x=dataset.x, y=dataset.y, validation_data=validation_data, test_data=test_data,
                    data_summary=dataset.dataset_name, iterations=iterations, epochs=epochs, batch_size=batch_size, polar=polar)


if __name__ == '__main__':
    print("Loading Dataset")
    T, labels = get_dataset_for_classification()
    T, labels = randomize(T, labels)
    x_train, y_train, x_test, y_test = separate_train_test_pixels(T, labels, ratio=0.1)
    x_train, y_train, x_val, y_val = separate_train_test_pixels(x_train, y_train, ratio=0.8)
    y_train = Dataset.sparse_into_categorical(y_train)
    y_test = Dataset.sparse_into_categorical(y_test)
    y_val = Dataset.sparse_into_categorical(y_val)
    dataset = Dataset(x_train.astype(np.complex64), y_train, dataset_name='Oberpfaffenhofen')
    print("Training model")
    run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
              test_data=(x_test.astype(np.complex64), y_test),
              iterations=50, epochs=300, batch_size=100,
              shape_raw=[100, 50], dropout=0.5, activation='cart_relu',
              polar=False)
    """
    shapes = [
        # [],
        # [5],
        [10], [50],
        [10, 10], [50, 50],
        [100, 50], [200, 100],
        [500], [500, 500],
        [1000]
        # [5000],
        # [500, 500], [5000, 5000],
        # [10000]
    ]
    for shape in shapes:
        run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
                  iterations=10, epochs=300,
                  optimizer='sgd', shape_raw=shape, activation='cart_relu', polar=False, dropout=None)"""
    """
    optimizers = [# 'sgd'
                  'rmsprop'
                  # cvnn.optimizers.SGD(learning_rate=0.1), cvnn.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  # cvnn.optimizers.SGD(learning_rate=0.01, momentum=0.75),
                  # cvnn.optimizers.RMSprop(learning_rate=0.001, rho=0.999, momentum=0.9),
                  # cvnn.optimizers.RMSprop(learning_rate=0.001, rho=0.8, momentum=0.75)]
                  ]
    functions = [# 'cart_relu',
                 'cart_tanh', 'cart_sigmoid']
    polar_modes = [
                   False
                   # True
    ]
    for opt in optimizers:
        for pol in polar_modes:
            for act_fun in functions:
                for shape in shapes:
                    run_monte(dataset, validation_data=(x_val.astype(np.complex64), y_val),
                              iterations=100, epochs=300,
                              optimizer=opt, shape_raw=shape, activation=act_fun, polar=pol)"""
