from cvnn.montecarlo import run_gaussian_dataset_montecarlo, get_mlp, run_montecarlo
from cvnn.data_analysis import MonteCarloAnalyzer
from notify_run import Notify
import numpy as np
from pdb import set_trace


def run_simulation():
    # notify = Notify()
    # notify.send('New simulation started')
    param_list = [
        [0.15, 1, 1],
        [-0.15, 1, 1]
    ]
    n = 128
    shapes = list(np.linspace(512, 4096, 10).astype(int))
    shapes = [16, 32, 64]
    # set_trace()
    models = []
    for i, sh in enumerate(shapes):
        models.append(get_mlp(input_size=n, output_size=2, shape_raw=[sh], dropout=0.5, name=f"{i}_{sh}"))
        # Should I use dropout or not?
    # run_gaussian_dataset_montecarlo(models=models, n=n, epochs=500, iterations=5, early_stop=True, debug=False,
    #                                 param_list=param_list, shuffle=True)
    run_montecarlo(models=models, dataset=None,
                   open_dataset="/home/barrachina/Documents/onera/circularity/log/montecarlo/2021/09September/13Monday/run-12h55m42",
                   epochs=2, iterations=2, early_stop=True, shuffle=True)
    # notify.send('Simulation Done')


def plotter(path):
    monte = MonteCarloAnalyzer(path=path)
    # monte.box_plot()
    monte.do_all()


if __name__ == "__main__":
    run_simulation()
    # plotter("/home/barrachina/Documents/onera/circularity/log/montecarlo/2021/09September/15Wednesday/run-15h24m55/run_data.csv")
