import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from cvnn import layers
from cvnn.layers import Dense
from cvnn.cvnn_model import CvnnModel
from cvnn.utils import transform_to_real
import cvnn.initializers as init
from tensorflow.keras.losses import categorical_crossentropy
from scipy.signal import hilbert
from cvnn.dataset import Dataset
from cvnn.montecarlo import mlp_run_real_comparison_montecarlo
from pdb import set_trace


def get_model(name, input_size, output_size, weight_init=init.GlorotUniform(), shape_raw=None, dropout=None):
    layers.ComplexLayer.last_layer_output_dtype = None
    layers.ComplexLayer.last_layer_output_size = None
    if shape_raw is None:
        shape_raw = [100, 40]
    shape = [Dense(input_size=input_size, output_size=shape_raw[0], activation='cart_relu',
                   input_dtype=np.float32, weight_initializer=weight_init, dropout=dropout)]
    for i in range(1, len(shape_raw)):  # TODO: Support empty shape_raw (no hidden layers)
        shape.append(Dense(output_size=shape_raw[i], activation='cart_relu',
                           weight_initializer=weight_init, dropout=dropout))
    shape.append(Dense(output_size=output_size, activation='softmax_real', weight_initializer=weight_init))
    return CvnnModel(name=name, shape=shape, loss_fun=categorical_crossentropy, verbose=False, tensorboard=False)


def split_train_test():
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def load_dataset():
    feature_column_names = []
    for i in range(48):  # Dataset presents 48 features
        feature_column_names.append("Feature" + str(i + 1))
    feature_column_names.append("label")

    dataset = pd.read_csv('/media/barrachina/data/datasets/sensorless_drive_diagnosis/Sensorless_drive_diagnosis.txt',
                          sep=" ", header=None, names=feature_column_names)
    y = label_binarize(np.array(dataset["label"]), classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    x = dataset.iloc[:, 0:48]
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    x = scaler.fit_transform(x)
    x = hilbert(x.astype(np.float32))
    return x, y


def train_one_it(x_train, x_test, y_train, y_test):
    x_train = transform_to_real(x_train).astype(np.float32)
    x_test = transform_to_real(x_test).astype(np.float32)

    model = get_model("Real_model", input_size=x_train.shape[1], output_size=y_train.shape[1],
                      weight_init=init.GlorotUniform(), shape_raw=[50, 30], dropout=0.5)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=100)


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test = split_train_test()
    # train_one_it(x_train, x_test, y_train, y_test)
    x, y = load_dataset()
    dataset = Dataset(x, y)
    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=10, epochs=50, batch_size=100, shape_raw=[150, 130],
                                       dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=10, epochs=50, batch_size=100, shape_raw=[200],
                                       dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=10,
                                       epochs=50, batch_size=100,
                                       shape_raw=[100, 50], dropout=0.5)

