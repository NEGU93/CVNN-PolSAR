import numpy as np
import tensorflow as tf
import sys
import os
import pandas as pd
from pdb import set_trace
import cvnn.data_processing as dp
import cvnn.data_analysis as da
import cvnn.cvnn_v1_compat as cvnnv1
import cvnn.cvnn_model as cvnnv2
import cvnn.layers as layers

montecarlo_options = {'CVNNvsRVNN', 'TFversion'}


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


def run_mlp_v1(x_train, y_train, x_test, y_test, dtype=np.complex64, axis_legends=None,
               tensorboard=False, verbose=False, save_loss_acc=False, epochs=100):
    print("Running v1 test")
    if dtype != np.complex64 and dtype != np.float32:
        sys.exit("Unsupported data type " + str(dtype))
    name = "gilles_net_"
    if dtype == np.complex64:
        name = name + "complex"
    elif dtype == np.float32:
        name = name + "real"
    # Hyper-parameters
    input_size = x_train.shape[1]  # Size of input
    output_size = y_train.shape[1]  # Size of output
    h1_size = 25
    h2_size = 10
    if dtype == np.float32:  # if the network must be real
        input_size *= 2  # double the input size
        h1_size *= 2
        h2_size *= 2
        # TODO: shall I multiply the hidden layers as well?
        x_train, x_test = dp.get_real_train_and_test(x_train, x_test)  # and transform data to real

    # Network creation
    mlp = cvnnv1.Cvnn(name, automatic_restore=False, logging_level="INFO", tensorboard=tensorboard,
                      verbose=verbose, save_loss_acc=save_loss_acc)
    shape = [layers.ComplexDense(input_size=input_size, output_size=h1_size, activation='cart_selu',
                                 input_dtype=dtype, output_dtype=dtype),
             layers.ComplexDense(input_size=h1_size, output_size=h2_size, activation='cart_selu',
                                 input_dtype=dtype, output_dtype=dtype),
             layers.ComplexDense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                                 input_dtype=dtype, output_dtype=np.float32)]
    mlp.create_mlp_graph(tf.keras.losses.categorical_crossentropy, shape)
    mlp.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=100, display_freq=1000)
    # print(da.categorical_confusion_matrix(mlp.predict(x_test), y_test, axis_legends=axis_legends))
    mlp.plot_loss_and_acc()
    mlp.plot_loss_and_acc()
    return mlp.compute_loss(x_test, y_test), mlp.compute_accuracy(x_test, y_test)


def run_mlp(x_train, y_train, x_test, y_test, epochs=100, batch_size=100, dtype=np.complex64,
            v1=False, display_freq=1000, learning_rate=0.01):
    print("Running test")
    name = "gilles_net_"
    if dtype == np.complex64 or dtype == np.complex128:
        name = name + "complex"
    elif dtype == np.float32 or dtype == np.float64:
        name = name + "real"
    else:
        sys.exit("Error: Unknown dtype for data " + str(x_train.dtype))

    # Hyper-parameters
    input_size = x_train.shape[1]  # Size of input
    output_size = y_train.shape[1]  # Size of output
    h1_size = 25
    h2_size = 10
    if dtype == np.float32 or dtype == np.float64:  # if the network must be real
        input_size *= 2  # double the input size
        h1_size *= 2
        h2_size *= 2
        x_train, x_test = dp.get_real_train_and_test(x_train, x_test)  # and transform data to real

    if v1:
        mlp = cvnnv1.Cvnn(name, learning_rate=learning_rate, automatic_restore=False, logging_level="INFO", verbose=True)
        shape = [layers.ComplexDense(input_size=input_size, output_size=h1_size, activation='cart_selu',
                                     input_dtype=dtype, output_dtype=dtype),
                 layers.ComplexDense(input_size=h1_size, output_size=h2_size, activation='cart_selu',
                                     input_dtype=dtype, output_dtype=dtype),
                 layers.ComplexDense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                                     input_dtype=dtype, output_dtype=np.float32)]
        mlp.create_mlp_graph(tf.keras.losses.categorical_crossentropy, shape=shape)
        mlp.train(x_train.astype(dtype), y_train, x_test, y_test,
                  epochs=epochs, batch_size=batch_size, display_freq=display_freq)
        loss = mlp.compute_loss(x_test.astype(dtype), y_test)
        acc = mlp.compute_accuracy(x_test.astype(dtype), y_test)
    else:
        shape = [
            layers.ComplexDense(input_size=input_size, output_size=h1_size, activation='cart_selu',
                                input_dtype=dtype, output_dtype=dtype),
            layers.ComplexDense(input_size=h1_size, output_size=h2_size, activation='cart_selu',
                                input_dtype=dtype, output_dtype=dtype),
            layers.ComplexDense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                                input_dtype=dtype, output_dtype=np.float32)
        ]
        mlp = cvnnv2.CvnnModel(name, shape, tf.keras.losses.categorical_crossentropy)
        mlp.fit(x_train.astype(dtype), y_train.astype(np.float32),
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, display_freq=display_freq)
        loss, acc = mlp.evaluate(x_train.astype(dtype), y_train.astype(np.float32))
        print("Train loss: {0:.4f}, accuracy: {1:.2f}".format(loss, 100 * acc))
        loss, acc = mlp.evaluate(x_test.astype(dtype), y_test.astype(np.float32))
        print("Validation loss: {0:.4f}, accuracy: {1:.2f}".format(loss, 100*acc))
        # print(da.categorical_confusion_matrix(mlp.predict(x_test), y_test, axis_legends=axis_legends))
        # mlp.plot_loss_and_acc()
        # mlp.plot_loss_and_acc()
    print("Loss = {0:.4f}; Acc = {1:.2f}".format(loss, 100*acc))
    return loss, acc


def monte_carlo_real_vs_complex_comparison(x_train, y_train, x_test, y_test, iterations=10,
                                           path='./results/', filename='CVNNvsRVNN', display_freq=100):
    write = True
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + filename + '.csv'):
        write = False  # Not to write again the CVNN and all the headers.
    file = open(path + filename + '.csv', 'a')
    print("Writing results into " + path + filename + '.csv')
    if write:
        file.write("CVNN loss,CVNN acc,RVNN loss,RVNN acc\n")
    for i in range(iterations):
        print("Iteration: " + str(i) + "/" + str(iterations))
        # runs network
        # tf.keras.backend.clear_session()
        # tf.compat.v1.enable_eager_execution()
        cvloss, cvacc = run_mlp(x_train, y_train, x_test, y_test, dtype=np.complex64,
                                epochs=50, v1=False, display_freq=display_freq, learning_rate=0.01)  # Test complex one
        rvloss, rvacc = run_mlp(x_train, y_train, x_test, y_test, dtype=np.float32,
                                epochs=50, v1=False, display_freq=display_freq, learning_rate=0.01)  # Test real one

        # save result
        file.write(str(cvloss) + "," + str(cvacc) + "," + str(rvloss) + "," + str(rvacc) + "\n")
        file.flush()  # Not to lose the data if MC stops in the middle
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(file.fileno())  # http://docs.python.org/2/library/stdtypes.html#file.flush
    file.close()


def monte_carlo_v1_vs_v2_comparison(x_train, y_train, x_test, y_test, iterations=10, path='./results/'):
    files = ["v2.csv", "v1.csv"]
    v1 = False
    for filename in files:
        write = True
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(path + filename):
            write = False  # Not to write again the headers.
        file = open(path + filename, 'a')
        print("Writing results into " + path + filename)
        if write:
            file.write("v2 loss,v2 acc\n")
        for i in range(iterations):
            print("Iteration: " + str(i) + "/" + str(iterations))
            # runs network
            # tf.keras.backend.clear_session()
            # tf.compat.v1.enable_eager_execution()
            cvloss, cvacc = run_mlp(x_train, y_train, x_test, y_test, dtype=np.complex64,
                                    epochs=10, v1=v1, display_freq=100, learning_rate=0.01)  # Test complex one
            file.write(str(cvloss) + "," + str(cvacc) + "\n")
            file.flush()  # Not to lose the data if MC stops in the middle
            # typically the above line would do. however this is used to ensure that the file is written
            os.fsync(file.fileno())  # http://docs.python.org/2/library/stdtypes.html#file.flush
        file.close()
        v1 = True   # TODO: Horribly but ok...


def show_monte_carlo_results(data_name, showfig=False, axis_legends=None, montecarlo='TFversion'):
    if montecarlo in montecarlo_options:
        if montecarlo == 'CVNNvsRVNN':
            d_mean = da.get_loss_and_acc_means("./results/" + data_name.replace(".mat", ".csv"))
            d_std = da.get_loss_and_acc_std("./results/" + data_name.replace(".mat", ".csv"))
            da.plot_2_gaussian(d_mean['CVNN acc'], d_std['CVNN acc'], d_mean['RVNN acc'], d_std['RVNN acc'], 'CVNN',
                               'RVNN',
                               x_label='accuracy', loc='upper right',
                               title='RVNN vs CVNN accuracy for ' + os.path.splitext(data_name)[0] + ' dataset',
                               filename="./results/CVNNvsRVNN_accuracy_" + data_name.replace(".mat", ".png"),
                               showfig=showfig)
            da.plot_2_gaussian(d_mean['CVNN loss'], d_std['CVNN loss'], d_mean['RVNN loss'], d_std['RVNN loss'], 'CVNN',
                               'RVNN',
                               x_label='loss', loc='upper right',
                               title='RVNN vs CVNN loss for ' + os.path.splitext(data_name)[0] + ' dataset',
                               filename="./results/CVNNvsRVNN_loss_" + data_name.replace(".mat", ".png"),
                               showfig=showfig)
            path, file = os.path.split("./results/" + data_name.replace(".mat", ".csv"))
            data = pd.read_csv("./results/" + data_name.replace(".mat", ".csv"))
            da.plot_csv_histogram(filename="./results/" + data_name.replace(".mat", ".csv"),
                                  data1=data['CVNN acc'], data2=data['RVNN acc'],
                                  showfig=showfig, bins=None, library='seaborn')
            # da.plot_csv_histogram(filename="./results/" + data_name.replace(".mat", ".csv"),
            #                       showfig=showfig, bins=bins, library='seaborn')
        else:
            v2_mean = da.get_loss_and_acc_means("./results/v2.csv")
            v2_std = da.get_loss_and_acc_std("./results/v2.csv")
            v1_mean = da.get_loss_and_acc_means("./results/v1.csv")
            v1_std = da.get_loss_and_acc_std("./results/v1.csv")
            da.plot_2_gaussian(v2_mean['v2 acc'], v2_std['v2 acc'], v1_mean['v1 acc'], v1_std['v1 acc'], 'v2', 'v1',
                               x_label='accuracy', loc='upper right',
                               title='TFv1 vs TFv2 accuracy for ' + os.path.splitext(data_name)[0] + ' dataset',
                               filename="./results/accuracy_versions_" + data_name.replace(".mat", ".png"),
                               showfig=showfig)
            da.plot_2_gaussian(v2_mean['v2 acc'], v2_std['v2 acc'], v1_mean['v1 acc'], v1_std['v1 acc'], 'v2', 'v1',
                               x_label='accuracy', loc='upper right',
                               title='TFv1 vs TFv2 loss for ' + os.path.splitext(data_name)[0] + ' dataset',
                               filename="./results/loss_versions_" + data_name.replace(".mat", ".png"),
                               showfig=showfig)
            data1 = pd.read_csv("./results/v2.csv")
            data2 = pd.read_csv("./results/v1.csv")
            da.plot_csv_histogram(filename="./results/" + data_name.replace(".mat", ".csv"),
                                  data1=data1['v2 acc'], data2=data2['v1 acc'], name_d1="TFv2", name_d2="TFv1",
                                  showfig=showfig, bins=None, library='seaborn')


def train_monte_carlo(data_name, iterations=10, montecarlo='TFversion'):
    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = dp.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)

    if montecarlo in montecarlo_options:
        if montecarlo == 'TFversion':
            monte_carlo_v1_vs_v2_comparison(x_train, y_train, x_test, y_test, iterations=iterations)
        else:
            # Train networks
            monte_carlo_real_vs_complex_comparison(x_train, y_train, x_test, y_test,
                                                   iterations=iterations, filename=os.path.splitext(data_name)[0])
    print("Monte carlo finished")


if __name__ == '__main__':
    data_2chirps = "data_cnn1d.mat"
    data_all_classes = "data_cnn1dC.mat"
    data_2chirps_test = "data_cnn1dT.mat"

    data_name = data_2chirps_test
    train_monte_carlo(data_name, iterations=5, montecarlo='CVNNvsRVNN')

    # Show results
    # show_monte_carlo_results(data_name, True)

    """
    # gets data
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data(data_name)
    cat_ic = dp.sparse_into_categorical(ic, num_classes=len(types))  # TODO: make sparse crossentropy test
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(xx, cat_ic, pre_rand=True)
    # train net

    # v1 = False
    # cvloss_v1, cvacc_v1 = run_mlp_v1(x_train, y_train, x_test, y_test, epochs=100)
    # cvloss, cvacc = run_mlp(x_train, y_train, x_test, y_test, epochs=10, v1=v1, display_freq=100, learning_rate=0.01)
    # rvloss, rvacc = run_mlp(x_train, y_train, x_test, y_test, epochs=100, v1=v1, dtype=np.float32)
    """

