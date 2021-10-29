import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import traceback
from pandas import DataFrame
from os import makedirs
from pdb import set_trace
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Optional, List, Union, Tuple
from cvnn.utils import REAL_CAST_MODES, create_folder, transform_to_real_map_function
from dataset_reader import labels_to_rgb, COLORS
from Oberpfaffenhofen.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from San_Francisco.sf_data_reader import SanFranciscoDataset
from Bretigny_ONERA.bretigny_dataset import BretignyDataset
from cao_fcnn import get_cao_fcnn_model
from zhang_cnn import get_zhang_cnn_model
from own_unet import get_my_unet_model

EPOCHS = 1
DATASET_META = {
    "SF-AIRSAR": {"classes": 5, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-ALOS2": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-GF3": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-RISAT": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    "SF-RS2": {"classes": 5, "orientation": "vertical", "percentage": (0.8, 0.2)},
    "OBER": {"classes": 3, "orientation": "vertical", "percentage": (0.85, 0.15)},
    "BRETIGNY": {"classes": 4, "orientation": "horizontal", "percentage": (0.7, 0.15, 0.15)}
}

MODEL_META = {
    "cao": {"size": 128, "stride": 25, "pad": 0, "batch_size": 30, "data_augment": True,
            "percentage": (0.9, 0.1), "task": "segmentation"},
    "own": {"size": 128, "stride": 25, "pad": 0, "batch_size": 32, "data_augment": True,
            "percentage": (0.9, 0.1), "task": "segmentation"},
    "zhang": {"size": 12, "stride": 1, "pad": 6, "batch_size": 100, "data_augment": False,
              "percentage": (0.09, 0.01, 0.9), "task": "classification"}
}


def get_callbacks_list(early_stop, temp_path):
    tensorboard_callback = callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt', save_weights_only=True,
                                            verbose=0, save_best_only=True)
    callback_list = [tensorboard_callback, cp_callback]
    if early_stop:
        if isinstance(early_stop, int):
            patience = early_stop
        else:
            patience = 5
        callback_list.append(callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=False
        ))
    return callback_list


def parse_input():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset_method', nargs=1, default=["random"], type=str,
                        help='One of:\n\t- random (default): randomly select the train and val set\n'
                             '\t- separate: split first the image into sections and select the sets from there\n'
                             '\t- single_separated_image: as separate, but do not apply the slinding window operation '
                             '\n\t\t(no batches, only one image per set). \n\t\tOnly possible with segmentation models')
    parser.add_argument('--tensorflow', action='store_true', help='Use tensorflow library')
    parser.add_argument('--epochs', nargs=1, type=int, default=[EPOCHS], help='(int) epochs to be done')
    parser.add_argument('--model', nargs=1, type=str, default=["cao"],
                        help='deep model to be used. Options:\n' +
                             "".join([f"\t- {model}\n" for model in MODEL_META.keys()]))
    parser.add_argument('--early_stop', nargs='?', const=True, default=False, type=early_stop_type,
                        help='Apply early stopping to training')
    parser.add_argument('--balance', nargs=1, type=str, default=["None"], help='Deal with unbalanced dataset by:\n'
                                                                               '\t- loss: weighted loss\n'
                                                                               '\t- dataset: balance dataset by '
                                                                               'randomly remove pixels of '
                                                                               'predominant classes\n'
                                                                               '\t- any other string will be considered'
                                                                               ' as not balanced')
    parser.add_argument('--real_mode', type=str, nargs='?', const='real_imag', default='complex',
                        help='run real model instead of complex.\nIf [REAL_MODE] is used it should be one of:\n'
                             '\t- real_imag\n\t- amplitude_phase\n\t- amplitude_only\n\t- real_only')
    parser.add_argument('--coherency', action='store_true', help='Use coherency matrix instead of s')

    parser.add_argument("--dataset", nargs=1, type=str, default=["SF-AIRSAR"],
                        help="dataset to be used. Available options:\n" +
                             "".join([f"\t- {dataset}\n" for dataset in DATASET_META.keys()]))
    return parser.parse_args()


def early_stop_type(arg):
    if isinstance(arg, bool):
        return arg
    else:
        return int(arg)


def _get_dataset_handler(dataset_name: str, mode, complex_mode, real_mode, balance: bool, normalize: bool = False):
    dataset_name = dataset_name.upper()
    if dataset_name.startswith("SF"):
        dataset_handler = SanFranciscoDataset(dataset_name=dataset_name, mode=mode, balance_dataset=balance,
                                              complex_mode=complex_mode, real_mode=real_mode, normalize=normalize)
    elif dataset_name == "BRETIGNY":
        dataset_handler = BretignyDataset(mode=mode, complex_mode=complex_mode, real_mode=real_mode,
                                          normalize=normalize, balance_dataset=balance)
    elif dataset_name == "OBER":
        if mode != "t":
            raise ValueError(f"Oberfaffenhofen only supports data as coherency matrix (t). Asked for {mode}")
        dataset_handler = OberpfaffenhofenDataset(complex_mode=complex_mode, real_mode=real_mode, normalize=normalize,
                                                  balance_dataset=balance)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_handler


def _get_model(model_name: str, channels: int, weights: Optional[List[float]], real_mode: str, num_classes: int,
               complex_mode: bool = True, tensorflow: bool = False):
    model_name = model_name.lower()
    if complex_mode:
        name_prefix = "cv-"
        dtype = np.complex64
        if tensorflow:
            raise ValueError("Tensorflow library does not support complex mode")
    else:
        name_prefix = "rv-"
        dtype = np.float32
        channels = REAL_CAST_MODES[real_mode] * channels
    if model_name == "cao":
        model = get_cao_fcnn_model(input_shape=(None, None, channels), num_classes=num_classes,
                                   tensorflow=tensorflow,
                                   dtype=dtype, name=name_prefix + model_name, weights=weights)
    elif model_name == "own":
        model = get_my_unet_model(input_shape=(None, None, channels), num_classes=num_classes,
                                  tensorflow=tensorflow,
                                  dtype=dtype, name=name_prefix + model_name, weights=weights)
    elif model_name == "zhang":
        if weights is not None:
            print("WARNING: Zhang model does not support weighted loss")
        model = get_zhang_cnn_model(input_shape=(MODEL_META["zhang"]["size"], MODEL_META["zhang"]["size"], channels),
                                    num_classes=num_classes, tensorflow=tensorflow, dtype=dtype,
                                    name=name_prefix + model_name)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model


def open_saved_model(root_path, model_name: str, complex_mode: bool, weights, channels: int,
                     real_mode: str, tensorflow: bool, num_classes: int):
    model = _get_model(model_name=model_name, tensorflow=tensorflow,
                       channels=channels, weights=weights, real_mode=real_mode,
                       complex_mode=complex_mode, num_classes=num_classes)
    model.load_weights(str(root_path / "checkpoints/cp.ckpt"))
    return model


def save_result_image_from_saved_model(root_path, model_name: str,
                                       dataset_handler,  # dataset parameters
                                       weights, channels: int = 3,     # model hyper-parameters
                                       complex_mode: bool = True, real_mode: str = "real_imag",     # cv / rv format
                                       use_mask: bool = True, tensorflow: bool = False):
    full_image = dataset_handler.image
    seg = dataset_handler.labels
    if not complex_mode:
        full_image, seg = transform_to_real_map_function(full_image, seg)
    # I pad to make sure dimensions are Ok when downsampling and upsampling again.
    first_dim_pad = int(2 ** 5 * np.ceil(full_image.shape[0] / 2 ** 5)) - full_image.shape[0]
    second_dim_pad = int(2 ** 5 * np.ceil(full_image.shape[1] / 2 ** 5)) - full_image.shape[1]
    paddings = [
        [int(np.ceil(first_dim_pad / 2)), int(np.floor(first_dim_pad / 2))],
        [int(np.ceil(second_dim_pad / 2)), int(np.floor(second_dim_pad / 2))],
        [0, 0]
    ]
    if use_mask:
        mask = dataset_handler.sparse_labels
        mask = tf.pad(mask, paddings[:-1])
    else:
        mask = None
    full_image = tf.pad(full_image, paddings)
    full_image = tf.expand_dims(full_image, axis=0)   # add batch axis

    model = open_saved_model(root_path, model_name=model_name, complex_mode=complex_mode,
                             weights=weights, channels=channels, real_mode=real_mode,
                             tensorflow=tensorflow, num_classes=DATASET_META[dataset_handler.name]["classes"])
    prediction = model.predict(full_image)[0]
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    labels_to_rgb(prediction, savefig=str(root_path / "prediction"), mask=mask,
                  colors=COLORS[dataset_handler.name])


def run_model(model_name: str, balance: str, tensorflow: bool,
              mode: str, complex_mode: bool, real_mode: str,
              early_stop: Union[bool, int], epochs: int, temp_path,
              dataset_name: str, dataset_method: str, percentage: Optional[Union[Tuple[float], float]] = None,
              debug: bool = False):
    if percentage is None:
        if dataset_method == "random":
            percentage = MODEL_META[model_name]["percentage"]
        else:
            percentage = DATASET_META[dataset_name]["percentage"]
    # Dataset
    dataset_name = dataset_name.upper()
    mode = mode.lower()
    dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode,
                                           complex_mode=complex_mode, real_mode=real_mode, normalize=False,
                                           balance=(balance == "dataset"))
    ds_list = dataset_handler.get_dataset(method=dataset_method, task=MODEL_META[model_name]["task"],
                                          percentage=percentage,
                                          size=MODEL_META[model_name]["size"], stride=MODEL_META[model_name]["stride"],
                                          pad=MODEL_META[model_name]["pad"],
                                          shuffle=True, savefig=str(temp_path / "image_") if debug else None,
                                          orientation=DATASET_META[dataset_name]['orientation'],
                                          data_augment=False,
                                          batch_size=MODEL_META[model_name]['batch_size']
                                          )
    train_ds = ds_list[0]
    if len(ds_list) > 1:
        val_ds = ds_list[1]
    else:
        val_ds = None
    if len(ds_list) > 2:
        test_ds = ds_list[2]
    else:
        test_ds = None
    if debug:
        dataset_handler.print_ground_truth(path=temp_path)
    # Model
    weights = dataset_handler.weights
    model = _get_model(model_name=model_name,
                       channels=3 if mode == "s" else 6,
                       weights=weights if balance == "loss" else None,
                       real_mode=real_mode, num_classes=DATASET_META[dataset_name]["classes"],
                       complex_mode=complex_mode, tensorflow=tensorflow)
    callbacks = get_callbacks_list(early_stop, temp_path)
    # Training
    history = model.fit(x=train_ds, epochs=epochs,
                        validation_data=val_ds, shuffle=True, callbacks=callbacks)
    # Save results
    df = DataFrame.from_dict(history.history)
    return df, dataset_handler, weights


def run_wrapper(model_name: str, balance: str, tensorflow: bool,
                mode: str, complex_mode: bool, real_mode: str,
                early_stop: Union[int, bool], epochs: int,
                dataset_name: str, dataset_method: str,
                percentage: Optional[Union[Tuple[float], float]] = None, debug: bool = False):
    temp_path = create_folder("./log/")
    makedirs(temp_path, exist_ok=True)
    with open(temp_path / 'model_summary.txt', 'w+') as summary_file:
        summary_file.write(" ".join(sys.argv[1:]) + "\n")
        summary_file.write(f"Model: {'cv-' if complex_mode else 'rv-'}{model_name}\n")
        if not complex_mode:
            summary_file.write(f"\t{real_mode}\n")
        summary_file.write(f"Dataset: {dataset_name}:\t{mode}\n")
        summary_file.write(f"Other parameters:\n")
        summary_file.write(f"\tepochs: {epochs}\n")
        summary_file.write(f"\t{'' if early_stop else 'no'} early stop\n")
        summary_file.write(f"\tweighted {balance}\n")
    df, dataset_handler, weights = run_model(model_name=model_name, balance=balance, tensorflow=tensorflow,
                                             mode=mode, complex_mode=complex_mode, real_mode=real_mode,
                                             early_stop=early_stop, temp_path=temp_path, epochs=epochs,
                                             dataset_name=dataset_name, dataset_method=dataset_method,
                                             percentage=percentage, debug=debug)
    df.to_csv(str(temp_path / 'history_dict.csv'), index_label="epoch")
    if MODEL_META[model_name]["task"]:
        save_result_image_from_saved_model(temp_path, dataset_handler=dataset_handler, model_name=model_name,
                                           tensorflow=tensorflow, complex_mode=complex_mode, real_mode=real_mode,
                                           channels=3 if mode == "s" else 6, weights=weights)


if __name__ == "__main__":
    args = parse_input()
    run_wrapper(model_name=args.model[0], balance=args.balance[0], tensorflow=args.tensorflow,
                mode="t" if args.coherency else "s", complex_mode=True if args.real_mode == 'complex' else False,
                real_mode=args.real_mode, early_stop=args.early_stop, epochs=args.epochs[0],
                dataset_name=args.dataset[0], dataset_method=args.dataset_method[0], percentage=None)

