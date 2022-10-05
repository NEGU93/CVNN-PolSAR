import argparse
import itertools
import functools
from tqdm import tqdm
import logging
import os
import gc
import os.path
from argparse import RawTextHelpFormatter
from pathlib import Path
import sys
import socket
import traceback
import numpy as np
import pandas as pd
import time
from datetime import timedelta

try:
    from notify_run import Notify

    if socket.gethostname() == 'barrachina-SONDRA':  # My machine, I usually know what is happening here
        Notify = None
except ImportError:
    Notify = None
from pandas import DataFrame
from os import makedirs
from random import randint
from time import sleep

from tensorflow.keras import callbacks
import tensorflow as tf
from typing import Optional, List, Union, Tuple
from cvnn.utils import REAL_CAST_MODES, create_folder, transform_to_real_map_function
from cvnn.real_equiv_tools import EQUIV_TECHNIQUES
from dataset_reader import labels_to_rgb, COLORS, transform_to_real_with_numpy
from dataset_readers.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from dataset_readers.sf_data_reader import SanFranciscoDataset
from dataset_readers.bretigny_dataset import BretignyDataset
from dataset_readers.flevoland_data_reader import FlevolandDataset
from dataset_readers.garon_dataset import GaronDataset
from models.cao_fcnn import get_cao_fcnn_model
from models.zhang_cnn import get_zhang_cnn_model
from models.own_unet import get_my_unet_model, get_my_unet_tests
from models.haensch_mlp import get_haensch_mlp_model
from models.tan_3dcnn import get_tan_3d_cnn_model
from models.cnn_standard import get_cnn_model
from models.mlp_model import get_mlp_model
from models.small_unet import get_small_unet_model

from pdb import set_trace

if os.path.exists('/scratchm/jbarrach'):
    print("Running on Spiro ONERA")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

EPOCHS = 1
DROPOUT_DEFAULT = {
    "downsampling": None,
    "bottle_neck": None,
    "upsampling": None
}

DATASET_META = {
    "SF-AIRSAR": {"classes": 5, "azimuth": "vertical", "percentage": (0.8, 0.2)},
    # "SF-ALOS2": {"classes": 6, "azimuth": "vertical", "percentage": (0.8, 0.2)},
    # "SF-GF3": {"classes": 6, "azimuth": "vertical", "percentage": (0.8, 0.2)},
    # "SF-RISAT": {"classes": 6, "azimuth": "vertical", "percentage": (0.8, 0.2)},
    # "SF-RS2": {"classes": 5, "azimuth": "vertical", "percentage": (0.8, 0.2)},
    "OBER": {"classes": 3, "azimuth": "vertical", "percentage": (0.85, 0.15)},
    "FLEVOLAND": {"classes": 15, "azimuth": "horizontal", "percentage": (0.8, 0.1, 0.1)},
    "BRET": {"classes": 4, "azimuth": "horizontal", "percentage": (0.7, 0.15, 0.15)},
    "GARON": {"classes": 4, "azimuth": "vertical", "percentage": (0.7, 0.15, 0.15)}
}

MODEL_META = {
    "cao": {"size": 128, "stride": 25, "pad": 'same', "batch_size": 30,
            "percentage": (0.8, 0.1, 0.1), "task": "segmentation"},
    "own": {"size": 128, "stride": 25, "pad": 'same', "batch_size": 30,
            "percentage": (0.8, 0.1, 0.1), "task": "segmentation"},
    "small-unet": {"size": 16, "stride": 1, "pad": 'same', "batch_size": 100,
                   "percentage": (0.8, 0.1, 0.1), "task": "segmentation"},
    "zhang": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 100,
              "percentage": (0.09, 0.01, 0.1, 0.8), "task": "classification"},
    "cnn": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 100,
            "percentage": (0.08, 0.02, 0.8), "task": "classification"},
    "expanded-cnn": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 100,
                     "percentage": (0.08, 0.02, 0.1), "task": "classification"},
    "haensch": {"size": 1, "stride": 1, "pad": 'same', "batch_size": 100,
                "percentage": (0.02, 0.08, 0.1, 0.8), "task": "classification"},
    "mlp": {"size": 1, "stride": 1, "pad": 'same', "batch_size": 100,
            "percentage": (0.08, 0.02, 0.1), "task": "classification"},
    "expanded-mlp": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 100,
                     "percentage": (0.08, 0.02, 0.8), "task": "classification"},
    "tan": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 64,
            "percentage": (0.09, 0.01, 0.1, 0.8), "task": "classification"}
}


def get_callbacks_list(early_stop, temp_path):
    tensorboard_callback = callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt', save_weights_only=True,
                                            verbose=0, save_best_only=True, monitor='val_loss')
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


def dropout_type(arg):
    if arg == 'None' or arg == 'none':
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
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset_method', nargs=1, default=["random"], type=str,
                        help='One of:\n\t- random (default): randomly select the train and val set\n'
                             '\t- separate: split first the image into sections and select the sets from there\n'
                             '\t- single_separated_image: as separate, but do not apply the slinding window operation '
                             '\n\t\t(no batches, only one image per set). \n\t\tOnly possible with segmentation models')
    parser.add_argument('--equiv_technique', nargs=1, default=["ratio_tp"], type=str,
                        help="Available options:\n" +
                             "".join([f"\t- {technique}\n" for technique in EQUIV_TECHNIQUES]))
    parser.add_argument('--tensorflow', action='store_true', help='Use tensorflow library')
    parser.add_argument('--epochs', nargs=1, type=int, default=[EPOCHS], help='(int) epochs to be done')
    parser.add_argument('--learning_rate', nargs=1, type=float, default=[None], help='(float) optimizer learning rate')
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
    parser.add_argument('--model_index', nargs=1, type=int, default=[None])
    parser.add_argument('--depth', nargs=1, type=int, default=[5])
    parser.add_argument('--real_mode', type=str, nargs='?', const='real_imag', default='complex',
                        help='run real model instead of complex.\nIf [REAL_MODE] is used it should be one of:\n'
                             '\t- real_imag\n\t- amplitude_phase\n\t- amplitude_only\n\t- real_only')
    parser.add_argument('--dropout', nargs=3, type=dropout_type, default=[None, None, None],
                        help='dropout rate to be used on '
                             'downsampling, bottle neck, upsampling sections (in order). '
                             'Example: `python main.py --dropout 0.1 None 0.3` will use 10%% dropout on the '
                             'downsampling part and 30%% on the upsamlpling part and no dropout on the bottle neck.')
    parser.add_argument('--coherency', type=int, nargs='?', default=0, const=1,
                        help='Use coherency matrix instead of s. '
                             '(Optional) followed by an integer stating the '
                             'boxcar size used for coherency matrix averaging.')
    parser.add_argument("--dataset", nargs=1, type=str, default=["SF-AIRSAR"],
                        help="dataset to be used. Available options:\n" +
                             "".join([f"\t- {dataset}\n" for dataset in DATASET_META.keys()]))
    return parser.parse_args()


def early_stop_type(arg):
    if isinstance(arg, bool):
        return arg
    else:
        return int(arg)


def _get_dataset_handler(dataset_name: str, mode, balance: bool = False, coh_kernel_size: int = 1):
    coh_kernel_size = int(coh_kernel_size)  # For back compat we make int(bool) so default kernel size = 1.
    dataset_name = dataset_name.upper()
    if dataset_name.startswith("SF"):
        dataset_handler = SanFranciscoDataset(dataset_name=dataset_name, mode=mode, coh_kernel_size=coh_kernel_size)
    elif dataset_name == "BRET":
        dataset_handler = BretignyDataset(mode=mode, balance_dataset=balance, coh_kernel_size=coh_kernel_size)
    elif dataset_name == "OBER":
        if mode != "t":
            raise ValueError(f"Oberfaffenhofen only supports data as coherency matrix (t). Asked for {mode}")
        dataset_handler = OberpfaffenhofenDataset(coh_kernel_size=coh_kernel_size)
    elif dataset_name == "FLEVOLAND":
        if mode != "t":
            raise ValueError(f"Flevoland 15 only supports data as coherency matrix (t). Asked for {mode}")
        dataset_handler = FlevolandDataset(coh_kernel_size=coh_kernel_size)
    elif dataset_name == "GARON":
        dataset_handler = GaronDataset(mode=mode, coh_kernel_size=coh_kernel_size)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_handler


def _get_model(model_name: str, channels: int, weights: Optional[List[float]], real_mode: str, num_classes: int,
               dropout, complex_mode: bool = True, tensorflow: bool = False, equiv_technique="ratio_tp",
               model_index: Optional = None, learning_rate: Optional[float] = None, depth: int = 5):
    model_name = model_name.lower()
    if equiv_technique != "ratio_tp" and model_name != "mlp":
        logging.warning(f"Equivalent technique requested {equiv_technique} but model ({model_name})"
                        f"is not mlp so it will not be applied.")
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
                                   tensorflow=tensorflow, dropout_dict=dropout,
                                   dtype=dtype, name=name_prefix + model_name, weights=weights)
    elif model_name == "cnn":
        model = get_cnn_model(input_shape=(MODEL_META["cnn"]["size"], MODEL_META["cnn"]["size"], channels),
                              num_classes=num_classes, tensorflow=tensorflow, dtype=dtype, weights=weights,
                              dropout=dropout["downsampling"], learning_rate=learning_rate,
                              name=name_prefix + model_name)
    elif model_name == "expanded-cnn":
        model = get_cnn_model(input_shape=(MODEL_META["cnn"]["size"], MODEL_META["cnn"]["size"], channels),
                              num_classes=num_classes, tensorflow=tensorflow, dtype=dtype, weights=weights,
                              dropout=dropout["downsampling"], learning_rate=learning_rate,
                              name=name_prefix + model_name, hyper_dict={'complex_filters': [6, 12, 24]})
    elif model_name == "own":
        model = get_my_unet_tests(index=model_index, depth=depth,
                                  input_shape=(None, None, channels), num_classes=num_classes,
                                  tensorflow=tensorflow, dropout_dict=dropout,
                                  dtype=dtype, name=name_prefix + model_name, weights=weights)
    elif model_name == "zhang":
        if weights is not None:
            print("WARNING: Zhang model does not support weighted loss")
        model = get_zhang_cnn_model(input_shape=(MODEL_META["zhang"]["size"], MODEL_META["zhang"]["size"], channels),
                                    num_classes=num_classes, tensorflow=tensorflow, dtype=dtype,
                                    dropout=dropout["downsampling"],
                                    name=name_prefix + model_name)
    elif model_name == 'haensch':
        if weights is not None:
            print("WARNING: Haensch model does not support weighted loss")
        model = get_haensch_mlp_model(input_shape=(MODEL_META["haensch"]["size"],
                                                   MODEL_META["haensch"]["size"], channels),
                                      num_classes=num_classes, tensorflow=tensorflow, dtype=dtype,
                                      dropout=dropout["downsampling"],
                                      name=name_prefix + model_name)
    elif model_name == 'mlp':
        model = get_mlp_model(input_shape=(MODEL_META["mlp"]["size"],
                                           MODEL_META["mlp"]["size"], channels),
                              num_classes=num_classes, tensorflow=tensorflow, dtype=dtype, weights=weights,
                              dropout=dropout["downsampling"], equiv_technique=equiv_technique,
                              name=equiv_technique.replace('_', '-') + '-' + name_prefix + model_name)
    elif model_name == 'expanded-mlp':
        model = get_mlp_model(input_shape=(MODEL_META["expanded-mlp"]["size"],
                                           MODEL_META["expanded-mlp"]["size"], channels),
                              num_classes=num_classes, tensorflow=tensorflow, dtype=dtype, weights=weights,
                              dropout=dropout["downsampling"], equiv_technique=equiv_technique,
                              name=equiv_technique.replace('_', '-') + '-' + name_prefix + model_name)
    elif model_name == 'tan':
        if weights is not None:
            print("WARNING: Tan model does not support weighted loss")
        model = get_tan_3d_cnn_model(input_shape=(MODEL_META["tan"]["size"],
                                                  MODEL_META["tan"]["size"], channels),
                                     num_classes=num_classes, tensorflow=tensorflow, dtype=dtype,
                                     name=name_prefix + model_name)
    elif model_name == 'small-unet':
        model = get_small_unet_model(input_shape=(None, None, channels), num_classes=num_classes,
                                     tensorflow=tensorflow, dropout_dict=dropout,
                                     dtype=dtype, name=name_prefix + model_name, weights=weights)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model


def open_saved_model(root_path, model_name: str, complex_mode: bool, weights, channels: int, dropout,
                     real_mode: str, tensorflow: bool, num_classes: int, equiv_technique: str, depth: int = 5,
                     model_index: Optional = None, learning_rate: Optional[float] = None):
    if isinstance(root_path, str):
        root_path = Path(root_path)
    model = _get_model(model_name=model_name, model_index=model_index, depth=depth,
                       channels=channels, learning_rate=learning_rate,
                       weights=weights, equiv_technique=equiv_technique,
                       real_mode=real_mode, num_classes=num_classes,
                       complex_mode=complex_mode, tensorflow=tensorflow, dropout=dropout)
    model.load_weights(str(root_path / "checkpoints/cp.ckpt")).expect_partial()
    return model


def _final_result_segmentation(root_path, use_mask, dataset_handler, model):
    full_image = dataset_handler.image
    seg = dataset_handler.labels
    if not model.input.dtype.is_complex:
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
    seg = tf.pad(seg, paddings)
    full_image = tf.expand_dims(full_image, axis=0)  # add batch axis
    seg = tf.expand_dims(seg, axis=0)
    prediction = model.predict(full_image)[0]
    if os.path.isfile(str(root_path / 'evaluate.csv')):
        evaluate = _eval_list_to_dict(model.evaluate(full_image, seg), model.metrics_names)
        eval_df = pd.read_csv(str(root_path / 'evaluate.csv'), index_col=0)
        eval_df = pd.concat([eval_df, DataFrame.from_dict({'full_set': evaluate})], axis=1)
        eval_df.to_csv(str(root_path / 'evaluate.csv'))
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    # dataset_handler.print_ground_truth(label=prediction, transparent_image=False, mask=mask,
    #                                    path=str(root_path / "prediction"))
    labels_to_rgb(prediction, savefig=str(root_path / "prediction"), mask=mask, colors=COLORS[dataset_handler.name])


def _final_result_classification(root_path, use_mask, dataset_handler, model, complex_mode, real_mode,
                                 batch_size: int = 10000):
    shape = model.input.shape[1:]
    generator = dataset_handler.apply_sliding_on_self_data(stride=1, size=shape[:-1],
                                                           pad="same", classification=True, add_unlabeled=True)
    prediction = None
    pbar = tqdm(total=int(np.ceil(np.prod(dataset_handler.shape) / batch_size)))
    while True:
        patches = [x for _, x in zip(range(batch_size), generator)]  # TODO: itertools.islice did not work. Why?
        tiles = np.array([x[0] for x in patches])
        labels = np.array([x[1] for x in patches])  # TODO: Horrible management
        if len(patches) == 0:
            break
        # tiles = dataset_handler.get_patches_image_from_point_and_self_image([tiles], size=shape[:-1], pad="same")
        if not complex_mode:
            tiles, _ = transform_to_real_with_numpy(tiles, None, mode=real_mode)
        if prediction is not None:
            prediction = np.concatenate((prediction, model.predict(tiles)))
            eval_result = model.evaluate(tiles, np.array(labels), verbose=0)
            if not np.isclose(eval_result[1], 0):
                evaluation = np.append(evaluation, np.expand_dims(eval_result, axis=0), axis=0)
        else:
            prediction = model.predict(tiles)
            evaluation = np.expand_dims(model.evaluate(tiles, np.array(labels), verbose=0), axis=0)
        pbar.update(1)
    pbar.close()
    if use_mask:
        mask = dataset_handler.get_sparse_labels()
    else:
        mask = None
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    image_prediction = tf.reshape(prediction,
                                  shape=tuple(dataset_handler.get_image().shape[:-1]) + (prediction.shape[-1],))
    labels_to_rgb(image_prediction, savefig=str(root_path / "prediction"), mask=mask,
                  colors=COLORS[dataset_handler.name])
    # TODO: Removed this value from results to save memory on run
    if os.path.isfile(str(root_path / 'evaluate.csv')):
        evaluate = np.mean(evaluation, axis=0)
        evaluate = _eval_list_to_dict(evaluate, model.metrics_names)
        eval_df = pd.read_csv(str(root_path / 'evaluate.csv'), index_col=0)
        eval_df = pd.concat([eval_df, DataFrame.from_dict({'full_set': evaluate})], axis=1)
        eval_df.to_csv(str(root_path / 'evaluate.csv'))
    # except ValueError as error:
    #     raise error
    # except:  # tf.python.framework.errors_impl.InternalError:
    #     print("Could not predict full image due to memory issues")
    #     return None
    # next_tile = tiles[:int(tiles.shape[0]/10)]
    # prediction = model.predict(next_tile)
    # for i in range(1, 10):     # compute in groups of 10 for memory problems.
    #     next_tile = tiles[int(i*tiles.shape[0]/10):int((i+1)*tiles.shape[0]/10)]
    #     pred = model.predict(next_tile)
    #     prediction = np.append(prediction, pred, axis=0)


def get_final_model_results(root_path, model_name: str,
                            dataset_handler, equiv_technique: str,
                            # mode: str, balance: str, dataset_name: str,
                            dropout, channels: int = 3,  # model hyper-parameters
                            complex_mode: bool = True, real_mode: str = "real_imag",  # cv / rv format
                            use_mask: bool = True, tensorflow: bool = False, depth: int = 5,
                            model_index: Optional = None):
    # dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode,
    #                                        complex_mode=complex_mode, real_mode=real_mode,
    #                                        balance=(balance == "dataset"),
    #                                        normalize=False, classification=MODEL_META[model_name]['task'])
    model = open_saved_model(root_path, model_name=model_name, complex_mode=complex_mode,
                             weights=None,  # I am not training, so no need to use weights in the loss function here
                             channels=channels, real_mode=real_mode, dropout=dropout, equiv_technique=equiv_technique,
                             tensorflow=tensorflow, num_classes=DATASET_META[dataset_handler.name]["classes"],
                             depth=depth, model_index=model_index)
    if MODEL_META[model_name]['task'] == 'segmentation':
        _final_result_segmentation(root_path=root_path, model=model, dataset_handler=dataset_handler, use_mask=use_mask)
    elif MODEL_META[model_name]['task'] == 'classification':
        _final_result_classification(root_path=root_path, model=model, dataset_handler=dataset_handler,
                                     use_mask=use_mask, complex_mode=complex_mode, real_mode=real_mode)
    else:
        raise ValueError(f"Unknown task {MODEL_META[model_name]['task']}")


def _eval_list_to_dict(evaluate, metrics):
    return_dict = {}
    for i, m in enumerate(metrics):
        return_dict[m] = evaluate[i]
    return return_dict


def _get_confusion_matrix(model, data, y_true, num_classes):
    if isinstance(data, tf.data.Dataset):
        data_list = []
        labels_list = []
        for x in data:
            data_list.append(x[0].numpy())
            labels_list.append(x[1].numpy())
        labels = np.concatenate(labels_list, axis=0)
        data = np.concatenate(data_list, axis=0)
    elif isinstance(data, np.ndarray):
        labels = y_true
    else:
        raise ValueError(f"y_true {y_true} format not supported")
    prediction = model.predict(data)
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        real_prediction = (np.real(prediction) + np.imag(prediction)) / 2.
    else:
        real_prediction = prediction
    real_flatten_prediction = np.reshape(real_prediction, newshape=[-1, num_classes])
    flatten_y_true = np.reshape(labels, newshape=[-1, num_classes])
    mask = np.invert(np.all(flatten_y_true == 0, axis=1))
    flatten_filtered_y_true = flatten_y_true[mask]  # tf.boolean_mask(flatten_y_true, mask)
    filtered_y_pred = real_flatten_prediction[mask]  # tf.boolean_mask(real_flatten_prediction, mask)
    sparse_flatten_filtered_y_true = np.argmax(filtered_y_pred, axis=-1)
    sparse_flatten_filtered_y_pred = np.argmax(flatten_filtered_y_true, axis=-1)
    conf = tf.math.confusion_matrix(labels=sparse_flatten_filtered_y_true, predictions=sparse_flatten_filtered_y_pred)
    conf_df = DataFrame(data=conf.numpy())
    conf_df['Total'] = conf_df.sum(axis=1)
    conf_df.loc['Total'] = conf_df.sum(axis=0)
    # one = model.evaluate(x=x_input, y=y_true, batch_size=30)
    # two = model.evaluate(ds)
    return conf_df


def run_model(model_name: str, balance: str, tensorflow: bool,
              mode: str, complex_mode: bool, real_mode: str, coh_kernel_size: int,
              early_stop: Union[bool, int], epochs: int, equiv_technique: str, temp_path, dropout,
              dataset_name: str, dataset_method: str, learning_rate: Optional[float] = None,
              percentage: Optional[Union[Tuple[float], float]] = None, model_index: Optional = None,
              debug: bool = False, use_tf_dataset=True, depth: int = 5):
    # If I use stride = 1 on random dataset method I get train and validation superposition, so avoid them
    # avoid_coincidences = MODEL_META[model_name]['task'] == "classification" and dataset_method == "random"
    avoid_coincidences = False
    if percentage is None:
        if dataset_method == "random":
            percentage = MODEL_META[model_name]["percentage"]
        else:
            percentage = DATASET_META[dataset_name]["percentage"]
    balance_dataset = (balance == "dataset",) * (len(percentage) - 1) + (False,)
    # Dataset
    dataset_name = dataset_name.upper()
    mode = mode.lower()
    dataset_handler = _get_dataset_handler(dataset_name=dataset_name, mode=mode, coh_kernel_size=coh_kernel_size)
    size = 2 ** (2 + depth) if MODEL_META[model_name]['task'] == "segmentation" else MODEL_META[model_name]["size"]
    ds_list = dataset_handler.get_dataset(method=dataset_method, percentage=percentage,
                                          balance_dataset=balance_dataset,
                                          complex_mode=complex_mode, real_mode=real_mode,
                                          size=size,
                                          stride=size if avoid_coincidences else MODEL_META[model_name]["stride"],
                                          pad=MODEL_META[model_name]["pad"],
                                          classification=MODEL_META[model_name]['task'] == 'classification',
                                          shuffle=True, savefig=str(temp_path / "image_") if debug else None,
                                          azimuth=DATASET_META[dataset_name]['azimuth'],
                                          data_augment=False, batch_size=MODEL_META[model_name]['batch_size'],
                                          cast_to_np=not use_tf_dataset
                                          )
    train_ds = ds_list[0]
    val_ds = ds_list[1]
    test_ds = ds_list[2] if len(ds_list) >= 3 else None
    # tf.config.list_physical_devices()
    # print(f"memory usage {tf.config.experimental.get_memory_info('GPU:0')['current'] / 10**9} GB")
    if debug:
        dataset_handler.print_ground_truth(path=temp_path)
    # Model

    weights = 1 / dataset_handler.labels_occurrences if balance == "loss" else None
    model = _get_model(model_name=model_name, model_index=model_index, depth=depth,
                       channels=6 if mode == "t" else 3, learning_rate=learning_rate,
                       weights=weights, equiv_technique=equiv_technique,
                       real_mode=real_mode, num_classes=DATASET_META[dataset_name]["classes"],
                       complex_mode=complex_mode, tensorflow=tensorflow, dropout=dropout)
    with open(temp_path / 'model_summary.txt', 'a') as summary_file:
        model.summary(print_fn=lambda x: summary_file.write(x + '\n'), line_length=200)
    callbacks = get_callbacks_list(early_stop, temp_path)
    # Training
    history = model.fit(x=train_ds[0] if not use_tf_dataset else train_ds,
                        y=train_ds[1] if not use_tf_dataset else None, epochs=epochs,
                        batch_size=MODEL_META[model_name]['batch_size'],
                        validation_data=val_ds, shuffle=True, callbacks=callbacks)
    # Saving history
    df = DataFrame.from_dict(history.history)
    df.to_csv(str(temp_path / 'history_dict.csv'), index_label="epoch")
    del model
    evaluate = {}
    evaluate = add_eval_and_conf_matrix(train_ds, evaluate, 'train',
                                        temp_path, model_name, complex_mode, weights, equiv_technique, mode, dropout,
                                        real_mode, tensorflow, dataset_name, use_tf_dataset,
                                        depth=depth, model_index=model_index)
    if val_ds:
        evaluate = add_eval_and_conf_matrix(val_ds, evaluate, 'val',
                                            temp_path, model_name, complex_mode, weights, equiv_technique, mode,
                                            dropout, real_mode, tensorflow, dataset_name, use_tf_dataset,
                                            depth=depth, model_index=model_index)
    if test_ds:
        evaluate = add_eval_and_conf_matrix(test_ds, evaluate, 'test',
                                            temp_path, model_name, complex_mode, weights, equiv_technique, mode,
                                            dropout, real_mode, tensorflow, dataset_name, use_tf_dataset,
                                            depth=depth, model_index=model_index)
    eval_df = DataFrame.from_dict(evaluate)
    eval_df.to_csv(str(temp_path / 'evaluate.csv'))     # Override if already exists
    # Create prediction image
    get_final_model_results(root_path=temp_path, model_name=model_name,
                            dataset_handler=dataset_handler, equiv_technique=equiv_technique,
                            dropout=dropout, channels=6 if mode == "t" else 3,  # model hyper-parameters
                            complex_mode=complex_mode, real_mode=real_mode,  # cv / rv format
                            use_mask=False, tensorflow=tensorflow,
                            depth=depth, model_index=model_index)
    return None


def add_eval(dataset, checkpoint_model, ds_set, use_tf_dataset, temp_path, evaluate):
    evaluate[ds_set] = _eval_list_to_dict(
        evaluate=checkpoint_model.evaluate(dataset[0] if not use_tf_dataset else dataset,
                                           dataset[1] if not use_tf_dataset else None),
        metrics=checkpoint_model.metrics_names)
    eval_df = DataFrame.from_dict(evaluate)
    eval_df.to_csv(str(temp_path / 'evaluate.csv'))  # Override if already exists
    return evaluate


def add_eval_and_conf_matrix(dataset, evaluate, ds_set,
                             temp_path, model_name, complex_mode, weights, equiv_technique, mode, dropout, real_mode,
                             tensorflow, dataset_name, use_tf_dataset, depth, model_index):
    checkpoint_model = clear_and_open_saved_model(temp_path, model_name=model_name, complex_mode=complex_mode,
                                                  weights=weights, equiv_technique=equiv_technique,
                                                  channels=3 if mode == "s" else 6, dropout=dropout,
                                                  real_mode=real_mode, tensorflow=tensorflow,
                                                  num_classes=DATASET_META[dataset_name]["classes"],
                                                  depth=depth, model_index=model_index)
    # Create confusion matrix
    test_confusion_matrix = _get_confusion_matrix(checkpoint_model,
                                                  dataset[0] if not use_tf_dataset else dataset,
                                                  dataset[1] if not use_tf_dataset else None,
                                                  DATASET_META[dataset_name]["classes"])
    test_confusion_matrix.to_csv(str(temp_path / f"{ds_set}_confusion_matrix.csv"))
    # add values to evaluate dictionary
    return add_eval(dataset=dataset, checkpoint_model=checkpoint_model, ds_set=ds_set, use_tf_dataset=use_tf_dataset,
                    temp_path=temp_path, evaluate=evaluate)


def clear_and_open_saved_model(*args, **kwargs):
    if tf.config.list_physical_devices('GPU'):
        print(f"Clearing "
              f"{tf.config.experimental.get_memory_info('GPU:0')['current'] / 10 ** 9:.3f} GB of GPU memory usage")
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"GPU memory usage {tf.config.experimental.get_memory_info('GPU:0')['current'] / 10 ** 9:.3f} GB")
    return open_saved_model(*args, **kwargs)


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


def run_wrapper(model_name: str, balance: str, tensorflow: bool,
                mode: str, complex_mode: bool, real_mode: str,
                early_stop: Union[int, bool], epochs: int, coh_kernel_size: int,
                dataset_name: str, dataset_method: str, dropout, equiv_technique: str,
                model_index: Optional = None, learning_rate=None,
                percentage: Optional[Union[Tuple[float], float]] = None, debug: bool = False, depth: int = 5):
    temp_path = create_folder("./log/")
    makedirs(temp_path, exist_ok=True)
    dropout = parse_dropout(dropout=dropout)
    with open(temp_path / 'model_summary.txt', 'w+') as summary_file:
        summary_file.write(" ".join(sys.argv[1:]) + "\n")
        summary_file.write(f"\tRun on {socket.gethostname()}\n")
    run_model(model_name=model_name, balance=balance, tensorflow=tensorflow,
              mode=mode, complex_mode=complex_mode, real_mode=real_mode,
              early_stop=early_stop, temp_path=temp_path, epochs=epochs,
              dataset_name=dataset_name, dataset_method=dataset_method, learning_rate=learning_rate,
              percentage=percentage, debug=debug, dropout=dropout, model_index=model_index,
              coh_kernel_size=coh_kernel_size, equiv_technique=equiv_technique, depth=depth)


if __name__ == "__main__":
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    args = parse_input()
    start_time = time.monotonic()
    if Notify is not None:
        notify = Notify()
        sleep(randint(1, 30))  # Wait between 1 sec and half a minute
        notify.send(f"{socket.gethostname()}: Running simulation with params {' '.join(sys.argv[1:])}")
    try:
        # TODO: BUG?! I use s instead of k
        run_wrapper(model_name=args.model[0], balance=args.balance[0], tensorflow=args.tensorflow,
                    mode="t" if args.coherency else "s", coh_kernel_size=args.coherency,
                    model_index=args.model_index[0], learning_rate=args.learning_rate[0],
                    complex_mode=True if args.real_mode == 'complex' else False,
                    real_mode=args.real_mode, early_stop=args.early_stop, epochs=args.epochs[0],
                    dataset_name=args.dataset[0], dataset_method=args.dataset_method[0], percentage=None,
                    dropout=args.dropout, equiv_technique=args.equiv_technique[0], depth=args.depth[0])
    except Exception as e:
        traceback.print_exc()
        if Notify is not None:
            notify.send(f"{socket.gethostname()}: {e}")
            raise e
    else:
        if Notify is not None:
            notify.send(
                f"{socket.gethostname()}: Simulation ended in {timedelta(seconds=time.monotonic() - start_time)}")
