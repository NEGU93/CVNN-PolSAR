from cvnn.montecarlo import mlp_run_real_comparison_montecarlo, get_mlp, run_montecarlo
from cvnn.dataset import Dataset
from dataset_reader import get_coh_data, get_k_data
from notify_run import Notify
import traceback
from pdb import set_trace


def test_activations():
    x_train, y_train, x_val, y_val = get_coh_data()
    dataset = Dataset(x=x_train, y=y_train)
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output

    models = []
    activations = ['zrelu', 'cart_relu', 'complex_cardioid', 'modrelu']
    for act in activations:
        models.append(get_mlp(input_size=input_size, output_size=output_size, activation=act, name=act))

    run_montecarlo(models=models, dataset=dataset,
                   iterations=10, epochs=150, batch_size=100, display_freq=1,
                   validation_data=(x_val, y_val),
                   debug=False, do_conf_mat=False, do_all=True, tensorboard=False, polar=None, plot_data=False)


def test_output_act():
    x_train, y_train, x_val, y_val = get_coh_data()
    dataset = Dataset(x=x_train, y=y_train)
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output

    models = []
    activations = ['softmax_real_with_abs', 'softmax_real_with_avg', 'softmax_real_with_mult',
                   'softmax_of_softmax_real_with_mult', 'softmax_of_softmax_real_with_avg', 'softmax_real_with_polar']
    for act in activations:
        models.append(get_mlp(input_size=input_size, output_size=output_size, output_activation=act, name=act))

    run_montecarlo(models=models, dataset=dataset,
                   iterations=10, epochs=150, batch_size=100, display_freq=1,
                   validation_data=(x_val, y_val),
                   debug=False, do_conf_mat=False, do_all=True, tensorboard=False, polar=None, plot_data=False)


def train(x_train, y_train, x_val, y_val):
    dataset = Dataset(x=x_train, y=y_train)
    mlp_run_real_comparison_montecarlo(dataset=dataset, iterations=30,
                                       epochs=150, batch_size=100, display_freq=1,
                                       optimizer='sgd', dropout=0.5, shape_raw=[100, 50], activation='cart_relu',
                                       polar=None,  # output_activation='softmax_real_with_avg',
                                       debug=False, do_all=True, shuffle=False, tensorboard=False, plot_data=False,
                                       validation_data=(x_val, y_val),
                                       capacity_equivalent=True, equiv_technique='ratio'
                                       )


if __name__ == "__main__":
    # https://notify.run/c/PGqsOzNQ1cSGdWM7
    notify = Notify()
    notify.send('New simulation started')
    try:
        notify.send('Simulating k data')
        x_train, y_train, x_val, y_val = get_k_data()
        train(x_train, y_train, x_val, y_val)
        notify.send('Simulating coh data')
        x_train, y_train, x_val, y_val = get_coh_data()
        train(x_train, y_train, x_val, y_val)
        notify.send(f"Simulations done")
    except Exception as e:
        notify.send("Error occurred")
        print(e)
        traceback.print_exc()
