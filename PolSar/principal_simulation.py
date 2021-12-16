import argparse
import os
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
except ImportError:
    Notify = None
from pandas import DataFrame
from os import makedirs
from tensorflow.keras import callbacks
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Optional, List, Union, Tuple
from cvnn.utils import REAL_CAST_MODES, create_folder, transform_to_real_map_function
from dataset_reader import labels_to_rgb, COLORS
from Oberpfaffenhofen.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from San_Francisco.sf_data_reader import SanFranciscoDataset
from Bretigny_ONERA.bretigny_dataset import BretignyDataset
from models.cao_fcnn import get_cao_fcnn_model
from models.zhang_cnn import get_zhang_cnn_model
from models.own_unet import get_my_unet_model
from models.haensch_mlp import get_haensch_mlp_model
from models.tan_3dcnn import get_tan_3d_cnn_model

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
    "SF-AIRSAR": {"classes": 5, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-ALOS2": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-GF3": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    # "SF-RISAT": {"classes": 6, "orientation": "vertical", "percentage": (0.8, 0.2)},
    "SF-RS2": {"classes": 5, "orientation": "vertical", "percentage": (0.8, 0.2)},
    "OBER": {"classes": 3, "orientation": "vertical", "percentage": (0.85, 0.15)},
    "BRET": {"classes": 4, "orientation": "horizontal", "percentage": (0.7, 0.15, 0.15)}
}

MODEL_META = {
    "cao": {"size": 128, "stride": 25, "pad": 0, "batch_size": 30,
            "percentage": (0.8, 0.1, 0.1), "task": "segmentation"},
    "own": {"size": 128, "stride": 25, "pad": 0, "batch_size": 32,
            "percentage": (0.8, 0.1, 0.1), "task": "segmentation"},
    "zhang": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 100,
              "percentage": (0.09, 0.01, 0.9), "task": "classification"},
    "haensch": {"size": 1, "stride": 1, "pad": 'same', "batch_size": 100,
                "percentage": (0.02, 0.08, 0.9), "task": "classification"},
    "tan": {"size": 12, "stride": 1, "pad": 'same', "batch_size": 64,
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
    parser.add_argument('--dropout', nargs=3, type=dropout_type, default=[None, None, None],
                        help='dropout rate to be used on '
                             'downsampling, bottle neck, upsampling sections (in order). '
                             'Example: `python main.py --dropout 0.1 None 0.3` will use 10%% dropout on the '
                             'downsampling part and 30%% on the upsamlpling part and no dropout on the bottle neck.')
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


def _get_dataset_handler(dataset_name: str, mode, complex_mode, real_mode, balance: bool, normalize: bool = False, classification: bool = False):
    dataset_name = dataset_name.upper()
    if dataset_name.startswith("SF"):
        dataset_handler = SanFranciscoDataset(dataset_name=dataset_name, mode=mode, balance_dataset=balance,
                                              complex_mode=complex_mode, real_mode=real_mode, normalize=normalize,
                                              classification=classification)
    elif dataset_name == "BRET":
        dataset_handler = BretignyDataset(mode=mode, complex_mode=complex_mode, real_mode=real_mode,
                                          normalize=normalize, balance_dataset=balance, classification=classification)
    elif dataset_name == "OBER":
        if mode != "t":
            raise ValueError(f"Oberfaffenhofen only supports data as coherency matrix (t). Asked for {mode}")
        dataset_handler = OberpfaffenhofenDataset(complex_mode=complex_mode, real_mode=real_mode, normalize=normalize,
                                                  balance_dataset=balance, classification=classification)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_handler


def _get_model(model_name: str, channels: int, weights: Optional[List[float]], real_mode: str, num_classes: int,
               dropout, complex_mode: bool = True, tensorflow: bool = False):
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
                                   tensorflow=tensorflow, dropout_dict=dropout,
                                   dtype=dtype, name=name_prefix + model_name, weights=weights)
    elif model_name == "own":
        model = get_my_unet_model(input_shape=(None, None, channels), num_classes=num_classes,
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
    elif model_name == 'tan':
        if weights is not None:
            print("WARNING: Tan model does not support weighted loss")
        model = get_tan_3d_cnn_model(input_shape=(MODEL_META["tan"]["size"],
                                                  MODEL_META["tan"]["size"], channels),
                                     num_classes=num_classes, tensorflow=tensorflow, dtype=dtype,
                                     name=name_prefix + model_name)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model


def open_saved_model(root_path, model_name: str, complex_mode: bool, weights, channels: int, dropout,
                     real_mode: str, tensorflow: bool, num_classes: int):
    if isinstance(root_path, str):
        root_path = Path(root_path)
    model = _get_model(model_name=model_name, tensorflow=tensorflow, dropout=dropout,
                       channels=channels, weights=weights, real_mode=real_mode,
                       complex_mode=complex_mode, num_classes=num_classes)
    model.load_weights(str(root_path / "checkpoints/cp.ckpt"))
    return model


def _final_result_segmentation(root_path, use_mask, dataset_handler, model):
    full_image = dataset_handler.get_image()
    seg = dataset_handler.get_labels()
    if not dataset_handler.complex_mode:
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
        mask = dataset_handler.get_sparse_labels()
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
    labels_to_rgb(prediction, savefig=str(root_path / "prediction"), mask=mask, colors=COLORS[dataset_handler.name])


def _final_result_classification(root_path, use_mask, dataset_handler, model):
    shape = model.input.shape[1:]
    stride = 1
    tiles, label_tiles = dataset_handler.apply_sliding(stride=stride, size=shape[:-1], pad="same", classification=True)
    # set_trace()
    if not dataset_handler.complex_mode:
        tiles, label_tiles = transform_to_real_map_function(tiles, label_tiles, dataset_handler.real_mode)
    if use_mask:
        mask = dataset_handler.get_sparse_labels()
    else:
        mask = None
    prediction = model.predict(tiles)
    if os.path.isfile(str(root_path / 'evaluate.csv')):
        evaluate = _eval_list_to_dict(model.evaluate(tiles, label_tiles), model.metrics_names)
        eval_df = pd.read_csv(str(root_path / 'evaluate.csv'), index_col=0)
        eval_df = pd.concat([eval_df, DataFrame.from_dict({'full_set': evaluate})], axis=1)
        eval_df.to_csv(str(root_path / 'evaluate.csv'))
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    # set_trace()
    image_prediction = tf.reshape(prediction,
                                  shape=tuple(dataset_handler.get_image().shape[:-1]) + (prediction.shape[-1],))
    labels_to_rgb(image_prediction, savefig=str(root_path / "prediction"), mask=mask,
                  colors=COLORS[dataset_handler.name])


def get_final_model_results(root_path, model_name: str,
                            dataset_handler,  # dataset parameters
                            dropout, channels: int = 3,  # model hyper-parameters
                            complex_mode: bool = True, real_mode: str = "real_imag",  # cv / rv format
                            use_mask: bool = True, tensorflow: bool = False):
    model = open_saved_model(root_path, model_name=model_name, complex_mode=complex_mode,
                             weights=None,  # I am not training, so no need to use weights in the loss function here
                             channels=channels, real_mode=real_mode, dropout=dropout,
                             tensorflow=tensorflow, num_classes=DATASET_META[dataset_handler.name]["classes"])
    if MODEL_META[model_name]['task'] == 'segmentation':
        _final_result_segmentation(root_path=root_path, model=model, dataset_handler=dataset_handler, use_mask=use_mask)
    elif MODEL_META[model_name]['task'] == 'classification':
        _final_result_classification(root_path=root_path, model=model, dataset_handler=dataset_handler,
                                     use_mask=use_mask)
    else:
        raise ValueError(f"Unknown task {MODEL_META[model_name]['task']}")


def _eval_list_to_dict(evaluate, metrics):
    return_dict = {}
    for i, m in enumerate(metrics):
        return_dict[m] = evaluate[i]
    return return_dict


def _get_confusion_matrix(ds, model, num_classes):
    # x_input, y_true = np.concatenate([x for x, y in ds], axis=0), np.concatenate([y for x, y in ds], axis=0)
    x_input, y_true = ds
    prediction = model.predict(x_input)
    if tf.dtypes.as_dtype(prediction.dtype).is_complex:
        real_prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    else:
        real_prediction = prediction
    real_flatten_prediction = tf.reshape(real_prediction, shape=[-1, num_classes])
    flatten_y_true = tf.reshape(y_true, shape=[-1, num_classes])
    mask = np.invert(np.all(flatten_y_true == 0, axis=1))
    flatten_filtered_y_true = tf.boolean_mask(flatten_y_true, mask)
    filtered_y_pred = tf.boolean_mask(real_flatten_prediction, mask)
    sparse_flatten_filtered_y_true = tf.argmax(filtered_y_pred, axis=-1)
    sparse_flatten_filtered_y_pred = tf.argmax(flatten_filtered_y_true, axis=-1)
    conf = tf.math.confusion_matrix(labels=sparse_flatten_filtered_y_true, predictions=sparse_flatten_filtered_y_pred)
    conf_df = DataFrame(data=conf.numpy())
    conf_df['Total'] = conf_df.sum(axis=1)
    conf_df.loc['Total'] = conf_df.sum(axis=0)
    # one = model.evaluate(x=x_input, y=y_true, batch_size=30)
    # two = model.evaluate(ds)
    # set_trace()
    return conf_df


def run_model(model_name: str, balance: str, tensorflow: bool,
              mode: str, complex_mode: bool, real_mode: str,
              early_stop: Union[bool, int], epochs: int, temp_path, dropout,
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
                                           balance=(balance == "dataset"),
                                           classification=MODEL_META[model_name]['task'] == 'classification')
    ds_list = dataset_handler.get_dataset(method=dataset_method,
                                          percentage=percentage,
                                          size=MODEL_META[model_name]["size"], stride=MODEL_META[model_name]["stride"],
                                          pad=MODEL_META[model_name]["pad"],
                                          shuffle=True, savefig=str(temp_path / "image_") if debug else None,
                                          orientation=DATASET_META[dataset_name]['orientation'],
                                          data_augment=False,
                                          batch_size=MODEL_META[model_name]['batch_size'], use_tf_dataset=False
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
    weights = dataset_handler.get_weights()
    model = _get_model(model_name=model_name,
                       channels=3 if mode == "s" else 6,  # TODO: isn't 'k' an option?
                       weights=weights if balance == "loss" else None,
                       real_mode=real_mode, num_classes=DATASET_META[dataset_name]["classes"],
                       complex_mode=complex_mode, tensorflow=tensorflow, dropout=dropout)
    callbacks = get_callbacks_list(early_stop, temp_path)
    # Training
    history = model.fit(x=train_ds[0], y=train_ds[1], epochs=epochs, batch_size=MODEL_META[model_name]['batch_size'],
                        validation_data=val_ds, shuffle=True, callbacks=callbacks)
    df = DataFrame.from_dict(history.history)
    # Get best model
    # checkpoint_model = model
    checkpoint_model = open_saved_model(temp_path, model_name=model_name, complex_mode=complex_mode,
                                        weights=weights if balance == "loss" else None,
                                        channels=3 if mode == "s" else 6, dropout=dropout, real_mode=real_mode,
                                        tensorflow=tensorflow, num_classes=DATASET_META[dataset_name]["classes"])
    evaluate = {'train': _eval_list_to_dict(evaluate=checkpoint_model.evaluate(train_ds[0], train_ds[1],
                                                                               batch_size=MODEL_META[model_name]['batch_size']),
                                            metrics=checkpoint_model.metrics_names)}
    train_confusion_matrix = _get_confusion_matrix(train_ds, checkpoint_model, DATASET_META[dataset_name]["classes"])
    train_confusion_matrix.to_csv(str(temp_path / 'train_confusion_matrix.csv'))
    if val_ds:
        evaluate['val'] = _eval_list_to_dict(evaluate=checkpoint_model.evaluate(val_ds[0], val_ds[1]),
                                             metrics=checkpoint_model.metrics_names)
        val_confusion_matrix = _get_confusion_matrix(val_ds, checkpoint_model, DATASET_META[dataset_name]["classes"])
        val_confusion_matrix.to_csv(str(temp_path / 'val_confusion_matrix.csv'))
    if test_ds:
        evaluate['test'] = _eval_list_to_dict(evaluate=checkpoint_model.evaluate(test_ds[0], test_ds[1]),
                                              metrics=checkpoint_model.metrics_names)
        test_confusion_matrix = _get_confusion_matrix(test_ds, checkpoint_model, DATASET_META[dataset_name]["classes"])
        test_confusion_matrix.to_csv(str(temp_path / 'test_confusion_matrix.csv'))
    eval_df = DataFrame.from_dict(evaluate)
    return df, dataset_handler, eval_df


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
                early_stop: Union[int, bool], epochs: int,
                dataset_name: str, dataset_method: str, dropout,
                percentage: Optional[Union[Tuple[float], float]] = None, debug: bool = False):
    temp_path = create_folder("./log/")
    makedirs(temp_path, exist_ok=True)
    dropout = parse_dropout(dropout=dropout)
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
    df, dataset_handler, eval_df = run_model(model_name=model_name, balance=balance, tensorflow=tensorflow,
                                                      mode=mode, complex_mode=complex_mode, real_mode=real_mode,
                                                      early_stop=early_stop, temp_path=temp_path, epochs=epochs,
                                                      dataset_name=dataset_name, dataset_method=dataset_method,
                                                      percentage=percentage, debug=debug, dropout=dropout)
    df.to_csv(str(temp_path / 'history_dict.csv'), index_label="epoch")
    eval_df.to_csv(str(temp_path / 'evaluate.csv'))
    get_final_model_results(temp_path, dataset_handler=dataset_handler, model_name=model_name,
                            tensorflow=tensorflow, complex_mode=complex_mode, real_mode=real_mode,
                            channels=3 if mode == "s" else 6, dropout=dropout)


if __name__ == "__main__":
    args = parse_input()
    start_time = time.monotonic()
    if Notify is not None:
        notify = Notify()
        notify.send(f"{socket.gethostname()}: Running simulation with params {' '.join(sys.argv[1:])}")
    try:
        run_wrapper(model_name=args.model[0], balance=args.balance[0], tensorflow=args.tensorflow,
                    mode="t" if args.coherency else "s", complex_mode=True if args.real_mode == 'complex' else False,
                    real_mode=args.real_mode, early_stop=args.early_stop, epochs=args.epochs[0],
                    dataset_name=args.dataset[0], dataset_method=args.dataset_method[0], percentage=None,
                    dropout=args.dropout)
    except Exception as e:
        if Notify is not None:
            notify.send(f"{socket.gethostname()}: {e}")
            traceback.print_exc()
    else:
        if Notify is not None:
            notify.send(f"{socket.gethostname()}: Simulation ended in {timedelta(seconds=time.monotonic() - start_time)}")
