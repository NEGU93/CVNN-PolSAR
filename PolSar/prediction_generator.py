from pathlib import Path
from pdb import set_trace
import tensorflow as tf
import numpy as np
import json
from cvnn.utils import transform_to_real_map_function
from dataset_reader import labels_to_rgb, COLORS
from principal_simulation import get_final_model_results, DROPOUT_DEFAULT, open_saved_model, DATASET_META
from dataset_readers.bretigny_dataset import BretignyDataset
from dataset_readers.flevoland_data_reader import FlevolandDataset
from dataset_readers.garon_dataset import GaronDataset
from results_reader import ResultReader


def bretigny_mask():
    path = Path("/home/barrachina/Documents/onera/PolSar/log/2022/05May/12Thursday/run-10h55m58")
    dataset_handler = BretignyDataset(mode="k", balance_dataset=True)
    get_final_model_results(path, dataset_handler=dataset_handler, model_name="cao",
                            tensorflow=True, complex_mode=False,
                            channels=3, dropout=DROPOUT_DEFAULT, use_mask=True)


def plot_no_mask_flev():
    keys = [
        '{"balance": "none", "dataset": "FLEVOLAND", "dataset_method": "random", '
        '"dataset_mode": "coh", "dtype": "complex", "library": "cvnn", '
        '"model": "cao"}',
        '{"balance": "none", "dataset": "FLEVOLAND", "dataset_method": '
        '"random", "dataset_mode": "coh", "dtype": "real_imag", "library": '
        '"tensorflow", "model": "cao"}',
    ]
    simulation_results = ResultReader(root_dir="/media/barrachina/data/results/new method")
    for key in keys:
        path = simulation_results.find_closest_to(key, dataset="test", key_to_find="median", metric="accuracy")
        dict_key = json.loads(key)
        dataset_handler = FlevolandDataset(complex_mode=dict_key["dtype"] == "complex",
                                           real_mode=dict_key["dtype"])
        model = open_saved_model(Path(path).parent,
                                 model_name=dict_key["model"], complex_mode=dict_key["dtype"] == "complex",
                                 weights=None,  # I am not training, so no need to use weights in the loss function here
                                 channels=6, real_mode=dict_key["dtype"], dropout=None,
                                 tensorflow=dict_key["library"] == "tensorflow",
                                 num_classes=DATASET_META[dataset_handler.name]["classes"])
        full_image = dataset_handler.get_image()
        seg = dataset_handler.get_labels()
        if not dataset_handler.complex_mode:
            full_image, seg = transform_to_real_map_function(full_image, seg)
        first_dim_pad = int(2 ** 5 * np.ceil(full_image.shape[0] / 2 ** 5)) - full_image.shape[0]
        second_dim_pad = int(2 ** 5 * np.ceil(full_image.shape[1] / 2 ** 5)) - full_image.shape[1]
        paddings = [
            [int(np.ceil(first_dim_pad / 2)), int(np.floor(first_dim_pad / 2))],
            [int(np.ceil(second_dim_pad / 2)), int(np.floor(second_dim_pad / 2))],
            [0, 0]
        ]
        full_image = tf.pad(full_image, paddings)
        full_image = tf.expand_dims(full_image, axis=0)  # add batch axis
        seg = tf.pad(seg, paddings)
        seg = tf.expand_dims(seg, axis=0)
        prediction = model.predict(full_image)[0]
        mask = dataset_handler.get_sparse_labels()
        mask = tf.pad(mask, paddings[:-1])
        coincidences = np.argmax(prediction, axis=-1) == np.argmax(seg, axis=-1)
        to_draw = np.array(np.stack([coincidences, np.invert(coincidences)], axis=-1), dtype=int)[0]
        # set_trace()
        labels_to_rgb(to_draw,
                      savefig=f"/home/barrachina/gretsi_images/{dict_key['dtype']}_{dict_key['model']}_prediction",
                      mask=mask, colors=[[255, 255, 255], [255, 0, 0]])


def garon():
    path = Path("/home/barrachina/Documents/onera/PolSar/log/2022/05May/03Tuesday/run-14h39m55")
    dataset_handler = GaronDataset(mode="s")
    get_final_model_results(path, dataset_handler=dataset_handler, model_name="cao",
                            tensorflow=True, complex_mode=False,
                            channels=3, dropout=DROPOUT_DEFAULT, use_mask=True)


if __name__ == "__main__":
    # garon()
    # plot_no_mask_flev()
    bretigny_mask()
