from dataset_reader import get_data
from cvnn.montecarlo import mlp_run_real_comparison_montecarlo
from cvnn.dataset import Dataset

if __name__ == "__main__":
    T, labels = get_data()
    flatten_T = T.reshape(shape=(T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
    dataset = Dataset(x=flatten_T, y=labels)
    assert flatten_T.shape[0] == 6 and len(flatten_T.shape) == 2
    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=30,
                                       epochs=300, batch_size=100, display_freq=1,
                                       optimizer='sgd', dropout=0.5, shape_raw=[100, 50], activation='cart_relu',
                                       polar=None,
                                       debug=False, do_all=True, shuffle=False, tensorboard=False, plot_data=False,
                                       validation_split=0.2, validation_data=None,
                                       capacity_equivalent=True, equiv_technique='ratio',
                                       )
