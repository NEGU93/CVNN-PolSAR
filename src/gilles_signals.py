import numpy as np
import tensorflow as tf
import sys
import os
from pdb import set_trace
import cvnn.data_processing as dp
import cvnn.data_analysis as da
import cvnn.cvnn_v1_compat as cvnnv1
import cvnn.cvnn_model as cvnnv2
import cvnn.layers as layers


def load_gilles_mat_data(fname, default_path="/media/barrachina/data/gilles_data/"):
    mat = dp.load_matlab_matrices(fname, default_path)
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


def run_mlp_v1(x_train, y_train, x_test, y_test, type=np.complex64, axis_legends=None,
               tensorboard=False, verbose=False, save_loss_acc=False, epochs=100):
    if type != np.complex64 and type != np.float32:
        sys.exit("Unsupported data type " + str(type))
    name = "gilles_net_"
    if type == np.complex64:
        name = name + "complex"
    elif type == np.float32:
        name = name + "real"
    # Hyper-parameters
    input_size = x_train.shape[1]  # Size of input
    output_size = y_train.shape[1]  # Size of output
    h1_size = 25
    h2_size = 10
    if type == np.float32:  # if the network must be real
        input_size *= 2  # double the input size
        h1_size *= 2
        h2_size *= 2
        # TODO: shall I multiply the hidden layers as well?
        x_train, x_test = dp.get_real_train_and_test(x_train, x_test)  # and transform data to real

    # Network creation
    mlp = cvnnv1.Cvnn(name, automatic_restore=False, logging_level="INFO", tensorboard=tensorboard,
                      verbose=verbose, save_loss_acc=save_loss_acc)
    mlp.create_mlp_graph(tf.keras.losses.categorical_crossentropy,
                         [layers.Dense(input_size=input_size, output_size=h1_size, activation='cart_selu',
                                       input_dtype=type, output_dtype=type),
                          layers.Dense(input_size=h1_size, output_size=h2_size, activation='cart_selu',
                                       input_dtype=type, output_dtype=type),
                          layers.Dense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                                       input_dtype=type, output_dtype=np.float32)])
    mlp.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=100, display_freq=1000)
    # print(da.categorical_confusion_matrix(mlp.predict(x_test), y_test, axis_legends=axis_legends))
    # mlp.plot_loss_and_acc()
    # mlp.plot_loss_and_acc()
    return mlp.compute_loss(x_test, y_test), mlp.compute_accuracy(x_test, y_test)


def run_mlp(x_train, y_train, x_test, y_test, axis_legends=None,
            tensorboard=False, verbose=False, save_loss_acc=False, epochs=100):
    name = "gilles_net_"
    if x_train.dtype == np.complex64 or x_train.dtype == np.complex128:
        name = name + "complex"
    elif x_train.dtype == np.float32 or x_train.dtype == np.float64:
        name = name + "real"
    else:
        sys.exit("Error: Unknown dtype for data " + str(x_train.dtype))

    # Hyper-parameters
    dtype = np.complex64
    input_size = x_train.shape[1]  # Size of input
    output_size = y_train.shape[1]  # Size of output
    h1_size = 25
    h2_size = 10
    if x_train.dtype == np.float32 or x_train.dtype == np.float64:  # if the network must be real
        input_size *= 2  # double the input size
        h1_size *= 2
        h2_size *= 2
        x_train, x_test = dp.get_real_train_and_test(x_train, x_test)  # and transform data to real

    shape = [layers.Dense(input_size=input_size, output_size=h1_size, activation='cart_sigmoid',
                          input_dtype=dtype, output_dtype=dtype),
             layers.Dense(input_size=h1_size, output_size=h2_size, activation='cart_sigmoid',
                          input_dtype=dtype, output_dtype=dtype),
             layers.Dense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                          input_dtype=dtype, output_dtype=np.float32)]
    # Network creation
    mlp = cvnnv2.CvnnModel(name, shape, tf.keras.losses.categorical_crossentropy, display_freq=1000)
    mlp.fit(x_train.astype(np.complex64), y_train, epochs=epochs, batch_size=100)
    # print(da.categorical_confusion_matrix(mlp.predict(x_test), y_test, axis_legends=axis_legends))
    # mlp.plot_loss_and_acc()
    # mlp.plot_loss_and_acc()
    return mlp.evaluate(x_test, y_test)


def monte_carlo_comparison(x_train, y_train, x_test, y_test, iterations=10, path='./results/', filename='CVNNvsRVNN'):
    write = True
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + filename + '.csv'):
        write = False  # Not to write again the CVNN and all the headers.
    file = open(path + filename + '.csv', 'a')
    if write:
        file.write("CVNN loss,CVNN acc,RVNN loss,RVNN acc\n")
    for i in range(iterations):
        print("Iteration: " + str(i) + "/" + str(iterations))
        # runs network
        cvloss, cvacc = run_mlp(x_train, y_train, x_test, y_test, np.complex64)  # Test complex one
        rvloss, rvacc = run_mlp(x_train, y_train, x_test, y_test, np.float32)  # Test real one

        # save result
        file.write(str(cvloss) + "," + str(cvacc) + "," + str(rvloss) + "," + str(rvacc) + "\n")
        file.flush()  # Not to lose the data if MC stops in the middle
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(file.fileno())  # http://docs.python.org/2/library/stdtypes.html#file.flush
    file.close()


def show_montecarlo_results(data_name, showfig=False, axis_legends=None):
    d_mean = da.get_loss_and_acc_means("./results/" + data_name.replace(".mat", ".csv"))
    d_std = da.get_loss_and_acc_std("./results/" + data_name.replace(".mat", ".csv"))
    da.plot_2_gaussian(d_mean['CVNN acc'], d_std['CVNN acc'], d_mean['RVNN acc'], d_std['RVNN acc'], 'CVNN', 'RVNN',
                       x_label='accuracy', loc='upper right',
                       title='RVNN vs CVNN accuracy for ' + os.path.splitext(data_name)[0] + ' dataset',
                       filename="./results/accuracy_" + data_name.replace(".mat", ".png"), showfig=showfig)
    da.plot_2_gaussian(d_mean['CVNN loss'], d_std['CVNN loss'], d_mean['RVNN loss'], d_std['RVNN loss'], 'CVNN', 'RVNN',
                       x_label='loss', loc='upper right',
                       title='RVNN vs CVNN loss for ' + os.path.splitext(data_name)[0] + ' dataset',
                       filename="./results/loss_" + data_name.replace(".mat", ".png"), showfig=showfig)
    da.plot_csv_histogram(filename="./results/" + data_name.replace(".mat", ".csv"),
                          showfig=showfig, bins=None, library='seaborn')
    # da.plot_csv_histogram(filename="./results/" + data_name.replace(".mat", ".csv"),
    #                       showfig=showfig, bins=bins, library='seaborn')


def train_monte_carlo(data_name, iterations=10):
    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = dp.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)

    # Train networks
    monte_carlo_comparison(x_train, y_train, x_test, y_test,
                           iterations=iterations, filename=os.path.splitext(data_name)[0])
    print("Monte carlo finished")


if __name__ == '__main__':
    data_2chirps = "data_cnn1d.mat"
    data_all_classes = "data_cnn1dC.mat"
    data_2chirps_test = "data_cnn1dT.mat"

    data_name = data_all_classes
    # train_monte_carlo(data_name, iterations=3000)

    # Show results
    # show_montecarlo_results(data_name, True)

    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = dp.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)
    # train net
    cvloss, cvacc = run_mlp(x_train, y_train, x_test, y_test, axis_legends=types,
                            tensorboard=True, verbose=True, save_loss_acc=True, epochs=200)
    rvloss, rvacc = run_mlp(x_train, y_train, x_test, y_test, axis_legends=types,
                            tensorboard=True, verbose=True, save_loss_acc=True, epochs=200)
