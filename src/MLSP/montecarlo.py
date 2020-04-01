import numpy as np
from cvnn.montecarlo import run_montecarlo


def first_simus():
    print("Running base case monte carlo")
    run_montecarlo()

    # Equal variance
    print("Running monte carlo with equal variance")
    param_list = [
        [0.5, 1, 1],
        [-0.5, 1, 1]
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
        run_montecarlo(activation=activation)

    # change m
    print("Running monte carlo for different sizes")
    per_class_examples = [5000, 2000, 1000, 500]  # 10000 already done
    for m in per_class_examples:
        run_montecarlo(m=m)

    # change learning rate TODO: RUN AGAIN!
    learning_rates = [0.001, 0.1, 1]        # 0.01 already done
    for learning_rate in learning_rates:
        run_montecarlo(learning_rate=learning_rate)


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


def shapes_simus():
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
        run_montecarlo(shape_raw=shape)

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


if __name__ == "__main__":
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