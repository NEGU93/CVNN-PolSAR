import numpy as np
from pdb import set_trace
from cvnn.dataset import Dataset
from cvnn.utils import load_matlab_matrices
from cvnn.montecarlo import run_montecarlo, mlp_run_real_comparison_montecarlo
from tensorflow.keras.losses import categorical_crossentropy
from cvnn.layers import Dense
from cvnn.cvnn_model import CvnnModel
from cvnn.data_analysis import SeveralMonteCarloComparison
import cvnn.initializers as init
import cvnn.layers as layers


def load_gilles_mat_data(fname, default_path="/media/barrachina/data/datasets/gilles_data"):
    mat = load_matlab_matrices(fname, default_path)
    ic = mat['ic'].squeeze(axis=0)  # Labels corresponding to types
    ic = ic - 1  # Changed from matlab indexing to python (start with 0 and not with 1)
    nb_sig = mat['nb_sig'].squeeze(axis=0)  # number of examples for each label (label=position_ic)
    sx = mat['sx'][0]  # Unknown scalar
    types = [t[0] for t in mat['types'].squeeze(axis=0)]  # labels legends
    xp = []  # Metadata TODO: good for regression network
    for t in mat['xp'].squeeze(axis=1):
        xp.append({'Type': t[0][0], 'Nb_rec': t[1][0][0], 'Amplitude': t[2][0][0], 'f0': t[3][0][0],
                   'Bande': t[4][0][0], 'Retard': t[5][0][0], 'Retard2': t[6][0][0], 'Sequence': t[7][0][0]})

    xx = mat['xx'].squeeze(axis=2).squeeze(axis=1).transpose()  # Signal data

    return ic, nb_sig, sx, types, xp, xx


def open_results():
    several = SeveralMonteCarloComparison('Gilles 7 classes data',
                                          x=["Old mode", "New version"],
                                          paths=[
                                              "/home/barrachina/Documents/onera/log/montecarlo/2020/07July/20Monday/run-10h00m17/run_data",
                                              "/home/barrachina/Documents/onera/log/montecarlo/2020/07July/21Tuesday/run-16h55m34/run_data"
                                          ]
                                          )
    several.box_plot(library='seaborn', savefile="./results/gilles_7_data", showfig=True)
    several.box_plot(library='plotly', savefile="./results/gilles_7_data", showfig=False)


def get_model(name, input_size, output_size, weight_init, shape_raw=None, dropout=None):
    layers.ComplexLayer.last_layer_output_dtype = None
    layers.ComplexLayer.last_layer_output_size = None
    if shape_raw is None:
        shape_raw = [100, 40]
    shape = [Dense(input_size=input_size, output_size=shape_raw[0], activation='cart_relu',
                   input_dtype=np.complex64, weight_initializer=weight_init, dropout=dropout)]
    for i in range(1, len(shape_raw)):  # TODO: Support empty shape_raw (no hidden layers)
        shape.append(Dense(output_size=shape_raw[i], activation='cart_relu',
                           weight_initializer=weight_init, dropout=dropout))
    shape.append(Dense(output_size=output_size, activation='softmax_real', weight_initializer=weight_init))
    return CvnnModel(name=name, shape=shape, loss_fun=categorical_crossentropy, verbose=False, tensorboard=False)


def run_gilles_data_on_specific_models(data_name, shape_raw, iterations=1000):
    # data_2chirps = "data_cnn1d.mat"
    # data_all_classes = "data_cnn1dC.mat"
    # data_2chirps_test = "data_cnn1dT.mat"

    #  data_name = data_all_classes
    cut_data = True
    if data_name == "data_cnn1dC.mat":
        cut_data = False

    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = Dataset.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x = xx.astype(np.complex64)
    y = cat_ic.astype(np.float32)
    if cut_data:
        y = y[:, 1:3]
    dataset = Dataset(x=x, y=y, dataset_name=data_name + "\n")
    # train net
    # montecarlo_analyzer = mlp_run_montecarlo(dataset=dataset, shape_raw=shape_raw)
    models = [
        get_model("GlorotUniform", input_size=dataset.x.shape[1], output_size=dataset.y.shape[1],
                  weight_init=init.GlorotUniform(), shape_raw=shape_raw),
        get_model("GlorotNormal", input_size=dataset.x.shape[1], output_size=dataset.y.shape[1],
                  weight_init=init.GlorotNormal(), shape_raw=shape_raw),
        get_model("GlorotUniformCompromise", input_size=dataset.x.shape[1], output_size=dataset.y.shape[1],
                  weight_init=init.GlorotUniformCompromise(), shape_raw=shape_raw),
        get_model("HeUniform", input_size=dataset.x.shape[1], output_size=dataset.y.shape[1],
                  weight_init=init.HeUniform(), shape_raw=shape_raw),
        get_model("HeNormal", input_size=dataset.x.shape[1], output_size=dataset.y.shape[1],
                  weight_init=init.HeNormal(), shape_raw=shape_raw),
    ]
    montecarlo_analyzer = run_montecarlo(dataset=dataset, models=models, iterations=iterations)


def run_gilles_data(data_name, shape_raw=None, dropout=0.5):
    if shape_raw is None:
        shape_raw = [100, 40]
    # data_name = data_all_classes
    cut_data = True     # Used to only have 2 classes and not 7 if only 2 are needed
    if data_name == "data_cnn1dC.mat":
        cut_data = False

    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = Dataset.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x = xx.astype(np.complex64)
    y = cat_ic.astype(np.float32)
    if cut_data:
        y = y[:, 1:3]
    dataset = Dataset(x=x, y=y, dataset_name=data_name + "\n")
    montecarlo_analyzer = mlp_run_real_comparison_montecarlo(dataset, open_dataset=None, iterations=500,
                                                             epochs=150, batch_size=100, display_freq=1,
                                                             learning_rate=0.01, shape_raw=shape_raw,
                                                             activation='cart_relu', debug=False, polar=False,
                                                             do_all=True, dropout=dropout, validation_split=0.2,
                                                             capacity_equivalent=True, equiv_technique='ratio')


if __name__ == '__main__':
    data_2chirps = "data_cnn1d.mat"
    data_all_classes = "data_cnn1dC.mat"
    data_2chirps_test = "data_cnn1dT.mat"
    run_gilles_data(data_all_classes, shape_raw=[120])
    run_gilles_data(data_2chirps, shape_raw=[120])
    run_gilles_data(data_all_classes, shape_raw=[120], dropout=None)
    run_gilles_data(data_2chirps, shape_raw=[120], dropout=None)
