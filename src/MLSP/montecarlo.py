import numpy as np
from cvnn.montecarlo import run_montecarlo


def first_simus(path="./data/MLSP"):
    print("Running base case monte carlo")
    run_montecarlo(open_dataset=path)

    # Equal variance
    print("Running monte carlo with both parameters to diferentiate")
    param_list = [
        [0.5, 2, 1],
        [-0.5, 1, 2]
    ]
    run_montecarlo(param_list=param_list)

    # No correlation
    print("Running monte carlo with no correlation")
    param_list = [[0, 1, 2], [0, 2, 1]]
    run_montecarlo(param_list=param_list)  # I have the theory polar will do bad here...

    # change activation function
    print("Running monte carlo for different activation functions")
    activation_function = ['cart_sigmoid', 'cart_tanh', 'cart_leaky_relu']  # relu already done
    for activation in activation_function:
        run_montecarlo(activation=activation, open_dataset=path)

    # change m
    print("Running monte carlo for different sizes")
    per_class_examples = [5000, 2000, 1000, 500]  # 10000 already done
    for m in per_class_examples:
        run_montecarlo(m=m)

    # change learning rate TODO: RUN AGAIN!
    learning_rates = [0.001, 0.1, 1]        # 0.01 already done
    for learning_rate in learning_rates:
        run_montecarlo(learning_rate=learning_rate, open_dataset=path)


def data_size():
    print("Run with different sizes of m but changing total epochs")
    # change m
    per_class_examples = [5000, 2000, 1000, 500]  # 10000 already done
    for m in per_class_examples:
        run_montecarlo(m=m, epochs=int(150*10000/m), batch_size=100)


def feature_vector_size():
    feature_vectors = [256, 64, 32, 16, 8]  # 128 already done
    for n in feature_vectors:
        run_montecarlo(n=n)


def multi_class_simus():
    # Multi Class
    print("Running multi-class (4) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 4)  # 4 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 2])
    run_montecarlo(param_list=param_list)
    print("Running multi-class (10) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 10)  # 10 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 2])
    run_montecarlo(param_list=param_list)


def shapes_simus(path="./data/MLSP"):
    # Single hidden layers
    shapes = [
        [512],
        [256],
        [128],
        [64],
        [32],
        [16],
        [8]
    ]
    for shape in shapes:
        print("Running for shape {}".format(shape[0]))
        run_montecarlo(shape_raw=shape, open_dataset=path)

    # Two hidden layers
    shapes = [
        [512, 100],
        [256, 50],
        [85, 40],
        [64, 32],
        [32, 50],
        [32, 80]
    ]
    for shape in shapes:
        print("Running for 2 hidden layers".format(shape[0]))
        run_montecarlo(shape_raw=shape, open_dataset=path)


def polar_simus():
    print("Running base case monte carlo")
    run_montecarlo(polar=True)

    # Equal variance
    print("Running monte carlo with equal variance")
    param_list = [
        [0.5, 1, 1],
        [-0.5, 1, 1]
    ]
    run_montecarlo(param_list=param_list, polar=True)

    # No correlation
    print("Running monte carlo with no correlation")
    param_list = [[0, 1, 2], [0, 2, 1]]
    run_montecarlo(param_list=param_list, polar=True)  # I have the theory polar will do bad here...


def hilando_fino():
    # Simuialtion of the 1st April 2020
    # Check if I can get the sigmoid better
    run_montecarlo(activation='cart_tanh', shape_raw=[256], learning_rate=0.01)
    # Check with not a lot of data
    run_montecarlo(shape_raw=[256], learning_rate=0.01, m=4000)
    run_montecarlo(shape_raw=[256], learning_rate=0.01, m=2000)
    run_montecarlo(shape_raw=[256], learning_rate=0.01, m=500)

    # Two hidden layers
    shapes = [
        [512, 100],
        [256, 50],
        [85, 40],
        [64, 32],
        [32, 50],
        [32, 80]
    ]
    for shape in shapes:
        print("Running for 2 hidden layers".format(shape[0]))
        run_montecarlo(shape_raw=shape)


def redo_sims_with_path():
    print("Redo simulations with dataset path")
    path = "./data/MLSP"
    run_montecarlo(open_dataset=path)

    # change activation function
    print("Running monte carlo for different activation functions")
    activation_function = ['cart_sigmoid', 'cart_tanh', 'cart_leaky_relu']  # relu already done
    for activation in activation_function:
        run_montecarlo(activation=activation, open_dataset=path)

    # learning rates
    learning_rates = [0.001, 0.1, 1]  # 0.01 already done
    for learning_rate in learning_rates:
        run_montecarlo(learning_rate=learning_rate, open_dataset=path)


if __name__ == "__main__":
    # Use both parameters to distinguish classes
    print("Use both parameters to distinguish classes")
    param_list = [
        [0.5, 2, 1],
        [-0.5, 1, 2]
    ]
    run_montecarlo(param_list=param_list)

    # Tanh activation function
    print("Tanh activation function")
    run_montecarlo(activation='cart_tanh', shape_raw=[64], learning_rate=0.1, param_list=[[0.5, 1, 2], [-0.5, 1, 2]])

    # data size with what Chengfang told me
    data_size()

    # change n
    feature_vector_size()

    # Use now the saved data
    # redo_sims_with_path()
