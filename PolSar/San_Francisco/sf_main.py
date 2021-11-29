import numpy as np
from os import makedirs
import sys
import argparse
from tensorflow.keras import callbacks
from tensorflow import pad, expand_dims, math
from pandas import DataFrame
import traceback
from typing import Optional, List
import os
from cvnn.utils import REAL_CAST_MODES, create_folder, transform_to_real_map_function
sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from models.cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model
from models.own_unet import get_my_unet_model
from dataset_reader import labels_to_rgb, SF_COLORS
from sf_data_reader import get_sf_cao_segmentation, open_image, get_labels, pauli_rgb_map_plot, get_sf_separated

EPOCHS = 100
NOTIFY = True
if NOTIFY:
    from notify_run import Notify
    notify = Notify()

DATASETS_CLASSES = {
    "AIRSAR": 5,
    "ALOS2": 6,
    "GF3": 6,
    "RISAT": 6,
    "RS2": 5
}


def get_callbacks_list(early_stop):
    temp_path = create_folder("./log/")
    makedirs(temp_path, exist_ok=True)
    tensorboard_callback = callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt', save_weights_only=True,
                                            verbose=0, save_best_only=True)
    callback_list = [tensorboard_callback, cp_callback]
    if early_stop:
        callback_list.append(callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=False
        ))
    return callback_list, temp_path


"""
    PARAM INPUT PARSERS
"""


def parse_dropout(dropout):
    if dropout is None:
        dropout = {
            "downsampling": None,
            "bottle_neck": None,
            "upsampling": None
        }
    elif isinstance(dropout, float):
        dropout = {
            "downsampling": dropout,
            "bottle_neck": dropout,
            "upsampling": dropout
        }
    elif isinstance(dropout, list):
        assert len(dropout) == 3, f"Dropout list should be of length 3, received {len(dropout)}"
        dropout = {
            "downsampling": dropout[0],
            "bottle_neck": dropout[1],
            "upsampling": dropout[2]
        }
    elif not isinstance(dropout, dict):
        raise ValueError(f"Unknown dataset format {dropout}")
    if "downsampling" not in dropout.keys():
        raise ValueError(f"downsampling should be a dropout key. dropout keys: {dropout.keys()}")
    if "bottle_neck" not in dropout.keys():
        raise ValueError(f"bottle_neck should be a dropout key. dropout keys: {dropout.keys()}")
    if "upsampling" not in dropout.keys():
        raise ValueError(f"upsampling should be a dropout key. dropout keys: {dropout.keys()}")
    return dropout


def dropout_type(arg):
    if arg == 'None':
        f = None
    else:
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f > 1. or f < 0.:
            raise argparse.ArgumentTypeError("Argument must be < " + str(1) + " and > " + str(0))
    return f


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--separate_dataset', action='store_true', help='Split dataset')
    parser.add_argument('--tensorflow', action='store_true', help='Use tensorflow library')
    parser.add_argument('--epochs', nargs=1, type=int, default=[EPOCHS], help='Epochs to be done')
    parser.add_argument('--model', nargs=1, type=str, default=["cao"], help='Deep model to be used. Options:\n\t- '
                                                                            'cao\n\town')
    parser.add_argument('--early_stop', action='store_true', help='Apply early stopping to training')
    parser.add_argument('--weighted_loss', action='store_true', help='Apply weights to loss for unbalanced datasets')
    parser.add_argument('--real_mode', type=str, nargs='?', const='real_imag', default='complex',
                        help='Run real model instead of complex')
    parser.add_argument('--coherency', action='store_true', help='Use coherency matrix instead of s')
    parser.add_argument('--dropout', nargs=3, type=dropout_type, default=[None, None, None],
                        help='Dropout rate to be used on '
                             'downsampling, bottle neck, upsampling sections (in order). '
                             'Example: `python main.py --dropout 0.1 None 0.3` will use 10%% dropout on the '
                             'downsampling part and 30%% on the upsamlpling part and no dropout on the bottle neck.')
    parser.add_argument("--dataset", nargs=1, type=str, default=["AIRSAR"], help="Dataset to be used. "
                                                                                 "Available options:\n\t- "
                                                                                 "AIRSAR\n\t- ALOS2\n\t- RS2")

    return parser.parse_args()


"""
    DATASET READER
"""


def open_saved_model(root_path, model_name: str, dataset: str, complex_mode: bool, weights, dropout, channels: int,
                     real_mode: str, tensorflow: bool):
    dropout = parse_dropout(dropout=dropout)
    model = _get_model(model_name=model_name, tensorflow=tensorflow,
                       channels=channels, dropout=dropout, weights=weights, real_mode=real_mode,
                       complex_mode=complex_mode, dataset=dataset)
    model.load_weights(str(root_path / "checkpoints/cp.ckpt"))
    return model


def save_result_image_from_saved_model(root_path, model_name: str,
                                       open_data: str, data_mode: str,  # dataset parameters
                                       weights, dropout=None, channels: int = 3,     # model hyper-parameters
                                       complex_mode: bool = True, real_mode: str = "real_imag",     # cv / rv format
                                       use_mask: bool = True, tensorflow: bool = False):
    full_image, seg = open_image(open_data=open_data, mode=data_mode)
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
        mask = get_labels(open_data=open_data)
        padded_mask = pad(mask, paddings[:-1])
    else:
        mask = None
        padded_mask = None
    pauli_rgb_map_plot(full_image, seg, dataset_name=open_data, path=root_path, mask=mask)
    full_image = pad(full_image, paddings)
    full_image = expand_dims(full_image, axis=0)   # add batch axis

    model = open_saved_model(root_path, model_name=model_name, dataset=open_data, complex_mode=complex_mode,
                             weights=weights, dropout=dropout, channels=channels, real_mode=real_mode,
                             tensorflow=tensorflow)
    prediction = model.predict(full_image)[0]
    prediction = (math.real(prediction) + math.imag(prediction)) / 2.
    labels_to_rgb(prediction, savefig=str(root_path / "prediction"), mask=padded_mask,
                  colors=SF_COLORS[open_data])


"""
    MODEL SIMULATION
"""


def _get_model(model_name: str, channels: int, dropout, weights: Optional[List[float]], real_mode: str, dataset: str,
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
        if not tensorflow:
            model = get_cao_cvfcn_model(input_shape=(None, None, channels), num_classes=DATASETS_CLASSES[dataset],
                                        dtype=dtype, name=name_prefix + model_name, dropout_dict=dropout, weights=weights)
        else:
            model = get_tf_real_cao_model(input_shape=(None, None, channels), num_classes=DATASETS_CLASSES[dataset],
                                          name=name_prefix + model_name, dropout_dict=dropout, weights=weights)
    elif model_name == "own":
        model = get_my_unet_model(input_shape=(None, None, channels), num_classes=DATASETS_CLASSES[dataset],
                                  tensorflow=tensorflow,
                                  dtype=dtype, name=name_prefix + model_name, dropout_dict=dropout, weights=weights)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model


def run_model(model_name: str, epochs: int, complex_mode: bool, coherency: bool, dropout: Optional[List[float]],
              early_stop: bool, dataset: str, weighted_loss: bool, tensorflow: bool = False,
              real_mode: str = 'real_imag', save_model: bool = True, separate_dataset: bool = False):
    try:
        dropout = parse_dropout(dropout=dropout)
        if not separate_dataset:
            train_dataset, test_dataset, weights = get_sf_cao_segmentation(open_data=dataset,
                                                                           mode="t" if coherency else "s",
                                                                           complex_mode=complex_mode,
                                                                           real_mode=real_mode)
        else:
            train_dataset, test_dataset, weights = get_sf_separated(open_data=dataset, mode="t" if coherency else "s",
                                                                    complex_mode=complex_mode, real_mode=real_mode)
        if not weighted_loss:
            weights = None
        channels = 6 if coherency else 3
        model = _get_model(model_name=model_name, channels=channels, dropout=dropout, weights=weights,
                           real_mode=real_mode, complex_mode=complex_mode, dataset=dataset, tensorflow=tensorflow)
        # Train
        callbacks, temp_path = get_callbacks_list(early_stop)
        with open(temp_path / 'model_summary.txt', 'w+') as summary_file:
            summary_file.write(" ".join(sys.argv[1:]) + "\n")
            summary_file.write(f"Model: {'cv-' if complex_mode else 'rv-'}{model_name}\n")
            if not complex_mode:
                summary_file.write(f"\t{real_mode}\n")
            summary_file.write(f"Dataset: {dataset}:\t{'coherency matrix' if coherency else 'raw data'}\n")
            summary_file.write(f"Other parameters:\n")
            summary_file.write(f"\t{epochs}\n")
            summary_file.write(f"\t{'' if early_stop else 'no'} early stop\n")
            summary_file.write(f"\t{'' if weighted_loss else 'no'} weighted loss\n")
        history = model.fit(x=train_dataset, epochs=epochs,
                            validation_data=test_dataset, shuffle=True, callbacks=callbacks)
        # Save results
        df = DataFrame.from_dict(history.history)
        df.to_csv(str(temp_path / 'history_dict.csv'), index_label="epoch")
        if NOTIFY:
            notify.send("Simulation done: " + " ".join(sys.argv[1:]))
        if save_model:
            save_result_image_from_saved_model(temp_path, model_name=model_name, tensorflow=tensorflow,
                                               open_data=dataset, data_mode="t" if coherency else "s",
                                               complex_mode=complex_mode, real_mode=real_mode,
                                               dropout=dropout, channels=channels, weights=weights)
        return None
    except Exception as e:
        if NOTIFY:
            notify.send(f"Error occurred: {e}")
        print(e)
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    args = parse_input()
    run_model(epochs=args.epochs[0], complex_mode=args.real_mode == 'complex', tensorflow=args.tensorflow,
              coherency=args.coherency, dropout=args.dropout, early_stop=args.early_stop, dataset=args.dataset[0],
              weighted_loss=args.weighted_loss, model_name=args.model[0], separate_dataset=args.separate_dataset)
    # save_result_image_from_saved_model(
    #     Path("/home/barrachina/Documents/onera/PolSar/San_Francisco/log/2021/10October/20Wednesday/run-21h43m54"),
    #     model_name=args.model[0],
    #     open_data=args.dataset[0], data_mode="t" if args.coherency else "s", weights=None,
    #     channels=6 if args.coherency else 3, complex_mode=args.real_mode == 'complex', real_mode=args.real_mode,
    #     use_mask=True
    # )

