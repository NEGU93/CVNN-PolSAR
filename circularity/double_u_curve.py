from cvnn.montecarlo import run_gaussian_dataset_montecarlo, get_mlp
from notify_run import Notify


if __name__ == "__main__":
    notify = Notify()
    notify.send('New simulation started')
    n = 128
    shapes = [
        [4],
        [8],
        [16],
        [32],
        [64],
        [128],
        [256],
        [512],
        [1024],
        [2048],
        [4096]
    ]
    models = []
    for sh in shapes:
        models.append(get_mlp(input_size=n, output_size=2, shape_raw=sh, dropout=None, name=f"{sh[0]}"))
        # Should I use dropout or not?
    run_gaussian_dataset_montecarlo(models=models, n=n, epochs=500, iterations=5, early_stop=True, debug=False)
    notify.send('Simulation Done')
