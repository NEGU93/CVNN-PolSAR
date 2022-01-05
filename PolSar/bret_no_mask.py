from pathlib import Path
from principal_simulation import get_final_model_results, DROPOUT_DEFAULT
from Bretigny_ONERA.bretigny_dataset import BretignyDataset


if __name__ == "__main__":
    path = Path("/media/barrachina/data/results/Bretigny/ICASSP_2022_trials/03Sunday/run-10h18m20")
    dataset_handler = BretignyDataset(mode="s", complex_mode=True, classification=False)
    get_final_model_results(path, dataset_handler=dataset_handler, model_name="cao",
                            tensorflow=False, complex_mode=True,
                            channels=3, dropout=DROPOUT_DEFAULT, use_mask=False)
