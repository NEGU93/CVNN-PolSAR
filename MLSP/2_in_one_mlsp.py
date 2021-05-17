import numpy as np
# from cvnn.montecarlo import run_montecarlo
from cvnn.montecarlo import run_gaussian_dataset_montecarlo, mlp_run_real_comparison_montecarlo
from cvnn.data_analysis import SeveralMonteCarloComparison
from cvnn.dataset import CorrelatedGaussianCoeffCorrel
from pdb import set_trace


def data_size(path, value, dropout=None):
    print("Run with different sizes of m but changing total epochs")
    # change m
    per_class_examples = [10000, 5000, 1500, 800]  # 10000 already done
    path_list = []
    for m in per_class_examples:
        if value == m:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(m=m, dropout=dropout, epochs=int(150*10000/m), batch_size=100))

    several = SeveralMonteCarloComparison("data_size",
                                          x=[str(i) for i in per_class_examples],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/data_size/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/data_size/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/data_size/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/data_size/several_train_loss_box_plot.html")


def feature_vector_size(path, value, dropout=None):
    print("Run with different feature vector sizes")
    feature_vectors = [256, 128, 64, 32, 16, 8]  # 128 already done
    path_list = []
    for n in feature_vectors:
        if n == value:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(n=n, dropout=dropout))

    several = SeveralMonteCarloComparison("feature_vector_size",
                                          x=[str(i) for i in feature_vectors],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/feature_vectors/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/feature_vectors/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/feature_vectors/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/feature_vectors/several_train_loss_box_plot.html")


def learning_rate(path, value, dropout=None):
    print("Run with different learning rates")
    feature_vectors = [0.0001, 0.001, 0.01, 0.1]  # 128 already done
    path_list = []
    for lr in feature_vectors:
        if value == lr:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(learning_rate=lr, dropout=dropout, open_dataset="./data/"))

    several = SeveralMonteCarloComparison("learning rate",
                                          x=[str(i) for i in feature_vectors],
                                          paths=path_list, round=4)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/learning_rate/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/learning_rate/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/learning_rate/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/learning_rate/several_train_loss_box_plot.html")


def coef_correl(path, value, dropout=None):
    coefs_list = [0.35, 0.5, 0.7, 0.99]
    path_list = []
    param_list = []
    for coef in coefs_list:
        param_list.append([[coef, 1, 1], [-coef, 1, 1]])
    for param in param_list:
        if param[0][0] == value:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(param_list=param, dropout=dropout))

    several = SeveralMonteCarloComparison("correlation coefficient",
                                          x=[str(i) for i in coefs_list],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/coef_correl/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/coef_correl/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/coef_correl/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/coef_correl/several_train_loss_box_plot.html")


def multi_class_simus(dropout=None):
    # Multi Class
    print("Running multi-class (4) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 4)  # 4 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 1])
    run_gaussian_dataset_montecarlo(param_list=param_list, dropout=dropout, do_all=True)
    print("Running multi-class (10) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 10)  # 10 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 1])
    run_gaussian_dataset_montecarlo(param_list=param_list, dropout=dropout, do_all=True)


def activation_function(path, value, dropout=None):
    print("Run with different activation functions")
    feature_vectors = ['cart_relu', 'cart_sigmoid', 'cart_tanh', 'cart_leaky_relu']  # 128 already done
    path_list = []
    for act in feature_vectors:
        if value == act:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(activation=act, dropout=dropout, open_dataset="./data/"))

    several = SeveralMonteCarloComparison("activation functions",
                                          x=[str(i) for i in feature_vectors],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/activation_functions/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/activation_functions/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/activation_functions/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/activation_functions/several_train_loss_box_plot.html")


"""def new_activation_function_analysis():
    print("Run with different activation functions")
    feature_vectors = []  # 128 already done
    path_list = []
    for act in feature_vectors:
        path_list.append(run_gaussian_dataset_montecarlo(activation='cart_tanh', open_dataset="./data/"))

    several = SeveralMonteCarloComparison("activation functions",
                                          x=[str(i) for i in feature_vectors],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/activation_functions/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/activation_functions/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/activation_functions/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/activation_functions/several_train_loss_box_plot.html")"""


def swipe_section():
    # path = run_montecarlo(open_dataset="./data/")
    path = "/home/barrachina/Documents/onera/src/MLSP/log/montecarlo/2020/04April/12Sunday/run-22h28m23/run_data.csv"
    run_gaussian_dataset_montecarlo(dropout=0.5, do_all=True)
    """
    print("Polar simulation")
    path_1 = run_montecarlo(polar=True, do_all=True, open_dataset="./data/")
    path_2 = run_montecarlo(param_list=[[0, 1, 2], [0, 2, 1]], polar=True, do_all=True)
    several = SeveralMonteCarloComparison("amplitude and phase", ["Type A", "Type B"], [path_1, path_2])
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/polar/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/polar/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/polar/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/polar/several_train_loss_box_plot.html")"""

    # run_montecarlo(param_list=[[0, 1, 2], [0, 2, 1]])       # Type B
    # run_montecarlo(param_list=[[0.5, 1, 2], [-0.5, 2, 1]])  # Type C

    # coef_correl(path=path, value=0.5)
    # activation_function(path=path, value='cart_relu')
    # learning_rate(path=path, value=0.01)
    # data_size(path=path, value=10000)
    # feature_vector_size(path=path, value=128)

    # multi_class_simus(dropout=0.5)

    # coef_correl(path=path, value=0.5, dropout=0.5)
    # activation_function(path=path, value='cart_relu', dropout=0.5)
    # learning_rate(path=path, value=0.01, dropout=0.5)
    # data_size(path=path, value=10000, dropout=0.5)
    # feature_vector_size(path=path, value=128, dropout=0.5)


def dropout_new_tests():
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeA/", polar=False, dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeA/", polar=True, dropout=0.5)
    # Type B
    print("Type B")
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeB/", dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeB/", shape_raw=[100, 40], dropout=0.5)
    # Type C
    print("Type C")
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeC/", dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeC/", shape_raw=[100, 40], dropout=0.5)

    print("Polar 2HL")
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeA/",
                                       polar=False, shape_raw=[100, 40], dropout=0.5)
    mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeA/",
                                       polar=True, shape_raw=[100, 40], dropout=0.5)


if __name__ == '__main__':
    """dataset = CorrelatedGaussianCoeffCorrel(m=10000, n=128, param_list=[[0.5, 1, 1], [-0.5, 1, 1]])
    dataset.save_data("./data/TypeA/")
    dataset = CorrelatedGaussianCoeffCorrel(m=10000, n=128, param_list=[[0., 2, 1], [0., 1, 2]])
    dataset.save_data("./data/TypeB/")
    dataset = CorrelatedGaussianCoeffCorrel(m=10000, n=128, param_list=[[0.5, 2, 1], [-0.5, 1, 2]])
    dataset.save_data("./data/TypeC/")"""
    # mlp_run_real_comparison_montecarlo(dataset=None, open_dataset="./data/TypeA/", iterations=10)
    dropout_new_tests()
