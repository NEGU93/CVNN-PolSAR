from cvnn.montecarlo import run_gaussian_dataset_montecarlo
from cvnn.data_analysis import SeveralMonteCarloComparison, MonteCarloAnalyzer
from pdb import set_trace
from tensorflow.keras.optimizers import SGD


def run():
    param_list = [
        [0.3, 1, 1],
        [-0.3, 1, 1]
    ]
    run_gaussian_dataset_montecarlo(m=10000, n=128, display_freq=1, optimizer=SGD(learning_rate=0.1),
                                    validation_split=0.2, activation='cart_relu', models=None,
                                    debug=False, polar=None, do_all=True, tensorboard=False, plot_data=False,
                                    epochs=300, batch_size=100, iterations=5,
                                    dropout=0.5, param_list=param_list, shape_raw=[64],
                                    capacity_equivalent=False, equiv_technique='ratio'
                                    )


if __name__ == "__main__":
    run()
