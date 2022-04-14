import sys
import os
from os import makedirs
from pathlib import Path
from pdb import set_trace
from cvnn.utils import create_folder

sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from principal_simulation import _get_dataset_handler, MODEL_META, DATASET_META


def test_dataset():
    model = "cao"
    for dataset in DATASET_META.keys():
        for mode in ["s", "t"]:
            for dataset_method in ["random", "separate", "single_separated_image"]:
                if not (mode == "s" and (dataset == "OBER" or dataset == "FLEV")):
                    print(f"Testing dataset {dataset} in mode {mode} with method {dataset_method}")
                    temp_path = Path(f"./log/{dataset}/{mode}/{dataset_method}/")
                    makedirs(temp_path, exist_ok=True)
                    if dataset_method == "random":
                        percentage = MODEL_META[model]["percentage"]
                    else:
                        percentage = DATASET_META[dataset]["percentage"]
                    dataset_handler = _get_dataset_handler(dataset_name=dataset, mode=mode, complex_mode=True,
                                                           real_mode="real_imag")
                    dataset_handler.get_dataset(method=dataset_method,
                                                percentage=percentage, size=MODEL_META[model]["size"],
                                                stride=MODEL_META[model]["stride"], pad=MODEL_META[model]["pad"],
                                                shuffle=True, savefig=None,     # str(temp_path / "sliced_"),
                                                orientation=DATASET_META[dataset]['orientation'],
                                                classification=True if MODEL_META[model]["task"] == "classification" else False,
                                                data_augment=False, batch_size=MODEL_META[model]['batch_size'])
                    dataset_handler.print_ground_truth()
                    dataset_handler.print_image_png(savefile=False, showfig=True)
                    del dataset_handler


test_dataset()
