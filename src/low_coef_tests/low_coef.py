from cvnn.dataset import Dataset, OpenDataset, get_parametric_predictor_labels, CorrelatedGaussianCoeffCorrel
from cvnn.montecarlo import mlp_run_real_comparison_montecarlo
from tensorflow.keras.optimizers import SGD
import numpy as np
from pdb import set_trace

# dataset = CorrelatedGaussianCoeffCorrel(m=10000, n=128, param_list=[[0.1, 1, 1], [-0.1, 1, 1]], sort=True)
# dataset.save_data("./ordered_data")

# set_trace()

dataset = OpenDataset("./ordered_data", 2)

x_train, y_train, x_test, y_test = dataset.get_train_and_test()
y_train_real = get_parametric_predictor_labels(x=np.real(x_train), y=np.imag(x_train), coef_1=0.1, coef_2=-0.1)
y_test_real = get_parametric_predictor_labels(x=np.real(x_test), y=np.imag(x_test), coef_1=0.1, coef_2=-0.1)
y_train_real = Dataset.sparse_into_categorical(y_train_real, 2)
y_test_real = Dataset.sparse_into_categorical(y_test_real, 2)

mlp_run_real_comparison_montecarlo(dataset=Dataset(x_train, y_train), validation_data=(x_test, y_test),
                                   optimizer=SGD(learning_rate=0.1), capacity_equivalent=False,
                                   shuffle=True, debug=False,
                                   iterations=30, epochs=300)
mlp_run_real_comparison_montecarlo(dataset=Dataset(x_train, y_train_real), validation_data=(x_test, y_test),
                                   optimizer=SGD(learning_rate=0.1), capacity_equivalent=False,
                                   shuffle=True, debug=False,
                                   iterations=30, epochs=300)
# mlp_run_real_comparison_montecarlo(dataset=Dataset(x_train, y_train_real), validation_data=(x_test, y_test_real),
#                                    optimizer=SGD(learning_rate=0.1), capacity_equivalent=False,
#                                    shuffle=True, debug=False,
#                                    iterations=30, epochs=300)

# set_trace()

