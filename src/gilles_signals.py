import data_processing as dp
import data_analysis as da
import cvnn_v1_compat as cvnn
import numpy as np
import sys
from pdb import set_trace


def load_gilles_mat_data(fname, default_path="/media/barrachina/data/gilles_data/"):
    mat = dp.load_matlab_matrices(fname, default_path)
    ic = mat['ic'].squeeze(axis=0)  # Labels corresponding to types
    nb_sig = mat['nb_sig'].squeeze(axis=0)  # number of examples for each label (label=position_ic)
    sx = mat['sx'][0]  # Unknown scalar
    types = [t[0] for t in mat['types'].squeeze(axis=0)]  # labels legends
    xp = []  # Metadata TODO: good for regression network
    for t in mat['xp'].squeeze(axis=1):
        xp.append({'Type': t[0][0], 'Nb_rec': t[1][0][0], 'Amplitude': t[2][0][0], 'f0': t[3][0][0],
                   'Bande': t[4][0][0], 'Retard': t[5][0][0], 'Retard2': t[6][0][0], 'Sequence': t[7][0][0]})

    xx = mat['xx'].squeeze(axis=2).squeeze(axis=1).transpose()  # Signal data

    return ic, nb_sig, sx, types, xp, xx


def run_mlp(x_train, y_train, x_test, y_test, type=np.complex64):
    if type != np.complex64 and type != np.float32:
        sys.exit("Unsupported data type " + str(type))
    # Hyper-parameters
    input_size = xx.shape[1]  # Size of input
    output_size = len(types)  # Number of classes
    h1_size = 25
    h2_size = 10
    if type == np.float32:  # if the network must be real
        input_size *= 2  # double the input size
        # TODO: shall I multiply the hidden layers as well?
        x_train, x_test = dp.get_real_train_and_test(x_train, x_test)  # and transform data to real

    # Network creation
    mlp = cvnn.Cvnn("Gilles_net_complex", automatic_restore=False, logging_level="INFO")
    mlp.create_mlp_graph("categorical_crossentropy",
                         [(input_size, 'ignored'),
                          (h1_size, 'cart_relu'),
                          (h2_size, 'cart_relu'),
                          (output_size, 'cart_softmax_real')],
                         input_dtype=type)

    mlp.train(x_train, y_train, x_test, y_test, epochs=100, batch_size=100, display_freq=1000)
    print(da.categorical_confusion_matrix(mlp.predict(x_test), y_test))
    mlp.plot_loss_and_acc()
    return mlp


if __name__ == '__main__':
    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data("data_cnn1dT.mat")
    cat_ic = dp.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)

    # runs network
    run_mlp(x_train, y_train, x_test, y_test, np.float32)     # Test complex one
    run_mlp(x_train, y_train, x_test, y_test, np.complex64)       # Test real one
    set_trace()
