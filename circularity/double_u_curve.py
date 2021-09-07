from cvnn.montecarlo import run_gaussian_dataset_montecarlo, get_mlp


if __name__ == "__main__":
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
        models.append(get_mlp(input_size=n, output_size=2, shape_raw=sh))
    run_gaussian_dataset_montecarlo(models=models, n=n, epochs=500, iterations=5, early_stop=True, debug=True)
