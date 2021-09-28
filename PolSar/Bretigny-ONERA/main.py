from pdb import set_trace
from os import path, makedirs
import sys
import numpy as np
import pickle
import tensorflow as tf
import argparse
import traceback
from cvnn.utils import create_folder
if path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
    NOTIFY = False
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
    NOTIFY = True
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar')
    NOTIFY = False
elif path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
    NOTIFY = False
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
if NOTIFY:
    from notify_run import Notify
from cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model
from bretigny_dataset import get_bret_cao_dataset, get_bret_separated_dataset
from image_generator import save_result_image_from_saved_model

EPOCHS = 100


"""----------
    Utils
----------"""


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxcar', nargs=1, type=int, default=[3], help='boxcar size for coherency matrix generation')
    parser.add_argument('--epochs', nargs=1, type=int, default=[EPOCHS], help='epochs to be done')
    parser.add_argument('--early_stop', action='store_true', help='apply early stopping to training')
    parser.add_argument('--complex', action='store_true', help='run complex model')
    parser.add_argument('--tensorflow', action='store_true', help='use tensorflow')
    parser.add_argument('--coherency', action='store_true', help='use coherency matrix instead of k')
    parser.add_argument('--split_datasets', action='store_true', help='Split the dataset into 3 parts to make sure '
                                                                      'train and test sets do not overlap')
    parser.add_argument('--dropout', nargs=3, type=dropout_type, default=[None, None, None],
                        help='dropout rate to be used on '
                             'downsampling, bottle neck, upsampling sections (in order). '
                             'Example: `python main.py --dropout 0.1 None 0.3` will use 10%% dropout on the '
                             'downsampling part and 30%% on the upsamlpling part and no dropout on the bottle neck.')

    return parser.parse_args()


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


def get_callbacks_list(early_stop):
    temp_path = create_folder("./log/")
    makedirs(temp_path, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=0, save_best_only=True)
    callback_list = [tensorboard_callback, cp_callback]
    if early_stop:
        callback_list.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=False
        ))
    return callback_list, temp_path


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


def run_model(epochs, complex_mode=True, tensorflow=False, dropout=None, coherency=False, split_datasets=False,
              save_model=True, early_stop=False, kernel_shape=3):
    try:
        if NOTIFY:
            notify = Notify()
            notify.send(f"Running Bretigny {'complex' if complex_mode else 'real'} model using "
                        f"{'cvnn' if not tensorflow else 'tf'} on {'coherency' if coherency else 'k'} data")
        dropout = parse_dropout(dropout=dropout)
        # Get dataset
        if split_datasets:
            train_dataset, test_dataset, _ = get_bret_separated_dataset(complex_mode=complex_mode, coherency=coherency,
                                                                        kernel_shape=kernel_shape)
        else:
            train_dataset, test_dataset = get_bret_cao_dataset(complex_mode=complex_mode, coherency=coherency,
                                                               kernel_shape=kernel_shape)
        channels = 6 if coherency else 3
        # Get model
        if not tensorflow:
            if complex_mode:
                model = get_cao_cvfcn_model(input_shape=(None, None, channels), num_classes=4,
                                            name="cao_cvfcn", dropout_dict=dropout)
            else:
                model = get_cao_cvfcn_model(input_shape=(None, None, 2*channels), num_classes=4,
                                            dtype=np.float32, name="cao_rvfcn", dropout_dict=dropout)
        else:
            if complex_mode:
                raise ValueError("Tensorflow does not support complex model. "
                                 "Do not use tensorflow and complex_mode both as True")
            model = get_tf_real_cao_model(input_shape=(None, None, 2*channels), num_classes=4,
                                          name="tf_cao_rvfcn", dropout_dict=dropout)
        # Train
        callbacks, temp_path = get_callbacks_list(early_stop)
        with open(temp_path / 'model_summary.txt', 'w+') as summary_file:
            summary_file.write(" ".join(sys.argv[1:]) + "\n")
            summary_file.write(f"Model Name: {model.name}\n")
            summary_file.write(f"Data type: {'complex' if complex_mode else 'real'}\n")
            summary_file.write(f"Library: {'cvnn' if not tensorflow else 'tensorflow'}\n")
            summary_file.write(f"Data format: {'coherency' if coherency else 'k'}\n")
            summary_file.write(f"\tdataset {'not' if not split_datasets else ''} splitted\n")
            summary_file.write(f"Dropout:\n")
            for key, value in dropout.items():
                summary_file.write(f"\t- {key}: {value}\n")
        history = model.fit(x=train_dataset, epochs=epochs,
                            validation_data=test_dataset, shuffle=True, callbacks=callbacks)
        if save_model:
            save_result_image_from_saved_model(temp_path, complex_mode=complex_mode, tensorflow=tensorflow,
                                               dropout=dropout, coherency=coherency)
        # Save results
        with open(temp_path / 'history_dict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        if NOTIFY:
            notify.send("Simulation done")
        return None
    except Exception as e:
        if NOTIFY:
            notify.send(f"Error occurred: {e}")
        print(e)
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    args = parse_input()
    run_model(epochs=args.epochs[0], complex_mode=args.complex, tensorflow=args.tensorflow, coherency=args.coherency,
              dropout=args.dropout, split_datasets=args.split_datasets, early_stop=args.early_stop,
              kernel_shape=args.boxcar[0])
