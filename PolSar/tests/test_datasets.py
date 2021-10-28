import sys
import os
from os import makedirs
from pathlib import Path
from cvnn.utils import create_folder

sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from principal_simulation import _get_dataset_handler, MODEL_META, DATASET_META


def test_dataset():
    model = "cao"
    for dataset in DATASET_META.keys():
        for mode in ["t", "s"]:
            for dataset_method in ["random", "separate", "single_separated_image"]:
                if not (mode == "s" and dataset == "OBER"):
                    temp_path = Path(f"./log/{dataset}/{mode}/{dataset_method}/")
                    makedirs(temp_path, exist_ok=True)
                    if dataset_method == "random":
                        percentage = MODEL_META[model]["percentage"]
                    else:
                        percentage = DATASET_META[dataset]["percentage"]
                    dataset_handler = _get_dataset_handler(dataset_name=dataset, mode=mode, normalize=False,
                                                           complex_mode=True, real_mode="real_imag")
                    dataset_handler.get_dataset(method=dataset_method, task=MODEL_META[model]["task"],
                                                percentage=percentage, size=MODEL_META[model]["size"],
                                                stride=MODEL_META[model]["stride"], pad=MODEL_META[model]["pad"],
                                                shuffle=True, savefig=str(temp_path / "sliced_"),
                                                orientation=DATASET_META[dataset]['orientation'],
                                                data_augment=False, batch_size=MODEL_META[model]['batch_size'])
                    dataset_handler.print_ground_truth(path=temp_path / "debug_image")


test_dataset()
