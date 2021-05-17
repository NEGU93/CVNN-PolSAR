from cvnn.montecarlo import run_gaussian_dataset_montecarlo, mlp_run_montecarlo
from cvnn.data_analysis import SeveralMonteCarloComparison


def data_size(path=None, value=None):
    print("Run with different sizes of m but changing total epochs")
    # change m
    per_class_examples = [10000, 5000, 1500, 800]  # 10000 already done
    path_list = []
    for m in per_class_examples:
        if value == m:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(m=m, epochs=int(150*10000/m), batch_size=100))

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


def coef_correl(path=None, value=None):
    print("Run with different sizes of m but changing total epochs")
    # change m
    per_class_examples = [0.3, 0.4, 0.5]  # 10000 already done
    path_list = []
    param_list = [
        [0, 1, 1],
        [0, 1, 1]
    ]
    for m in per_class_examples:
        if value == m:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(param_list=param_list))

    several = SeveralMonteCarloComparison("data_size",
                                          x=[str(i) for i in per_class_examples],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/coef_correl/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/coef_correl/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/coef_correl/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/coef_correl/several_train_loss_box_plot.html")


def coef_correl_2(path=None, value=None):
    print("Run with different sizes of m but changing total epochs")
    # change m
    per_class_examples = [0.3, 0.4, 0.5]  # 10000 already done
    path_list = []
    param_list = [
        [0, 1, 1],
        [0, 1, 1]
    ]
    for m in per_class_examples:
        if value == m:
            path_list.append(path)
        else:
            param_list[0][0] = m
            param_list[1][0] = -m
            path_list.append(run_gaussian_dataset_montecarlo(param_list=param_list, shape_raw=[100, 40]))

    several = SeveralMonteCarloComparison("data_size",
                                          x=[str(i) for i in per_class_examples],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/coef_correl_2HL/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/coef_correl_2HL/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/coef_correl_2HL/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/coef_correl_2HL/several_train_loss_box_plot.html")


def hidden_layer_size(path=None, value=None):
    print("Run with different feature vector sizes")
    feature_vectors = [16, 32, 64, 128, 256]  # 128 already done
    path_list = []
    for HL in feature_vectors:
        if HL == value:
            path_list.append(path)
        else:
            param_list[0][0] = m
            param_list[1][0] = -m
            path_list.append(run_gaussian_dataset_montecarlo(shape_raw=[HL]))

    several = SeveralMonteCarloComparison("feature_vector_size",
                                          x=[str(i) for i in feature_vectors],
                                          paths=path_list)
    several.box_plot(key='test accuracy', showfig=False,
                     savefile="./results/hidden_layer_size/several_test_accuracy_box_plot.html")
    several.box_plot(key='test loss', showfig=False,
                     savefile="./results/hidden_layer_size/several_test_loss_box_plot.html")
    several.box_plot(key='train accuracy', showfig=False,
                     savefile="./results/hidden_layer_size/several_train_accuracy_box_plot.html")
    several.box_plot(key='train loss', showfig=False,
                     savefile="./results/hidden_layer_size/several_train_loss_box_plot.html")


def feature_vector_size(path, value):
    print("Run with different feature vector sizes")
    feature_vectors = [256, 128, 64, 32, 16, 8]  # 128 already done
    path_list = []
    for n in feature_vectors:
        if n == value:
            path_list.append(path)
        else:
            path_list.append(run_gaussian_dataset_montecarlo(n=n))

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


param_list = [
    [0, 1, 2],
    [0, 2, 1]
]
run_gaussian_dataset_montecarlo(param_list=param_list)   # Dataset Type B
run_gaussian_dataset_montecarlo(param_list=param_list, shape_raw=[100, 40])
param_list = [
    [0.5, 1, 2],
    [-0.5, 2, 1]
]
run_gaussian_dataset_montecarlo(param_list=param_list)  # Dataset Type C
run_gaussian_dataset_montecarlo(param_list=param_list, shape_raw=[100, 40])

# Polar Mode
run_gaussian_dataset_montecarlo(polar=True)
run_gaussian_dataset_montecarlo(polar=True, shape_raw=[100, 40])

# Run base case for reference
base_path = run_gaussian_dataset_montecarlo()

hidden_layer_size(base_path, 64)
coef_correl(base_path, 0.5)
data_size(base_path, 10000)

coef_correl_2()


