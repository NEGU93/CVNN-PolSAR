import argparse
from pathlib import Path
from pdb import set_trace
import numpy as np
import tensorflow as tf
import os
import sys
if os.path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
elif os.path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar')
elif os.path.exists('/home/cfren/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/cfren/Documents/onera/PolSar')
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model
from bretigny_dataset import open_data, get_coherency_matrix, get_k_vector
from dataset_reader import labels_to_ground_truth
from cvnn.utils import transform_to_real_map_function


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


def read_parameters(root_path):
    with open(root_path / "model_summary.txt") as f:
        first_line = f.readline().rstrip().split(" ")
        if '--dropout' in first_line:
            indx = first_line.index("--dropout") + 1
            dropout = parse_dropout(first_line[indx:indx+3])
        else:
            dropout = [None, None, None]
        params_dict = {
            'complex': '--complex' in first_line,
            'tensorflow': '--tensorflow' in first_line,
            'coherency': '--coherency' in first_line,
            'split_datasets': '--split_datasets' in first_line,
            'dropout': dropout
        }
    return params_dict


def open_saved_model(root_path, complex_mode=True, tensorflow=False, dropout=None, coherency=False):
    dropout = parse_dropout(dropout=dropout)
    channels = 6 if coherency else 3
    if not tensorflow:
        if complex_mode:
            model = get_cao_cvfcn_model(input_shape=(None, None, channels), num_classes=4,
                                        name="cao_cvfcn", dropout_dict=dropout)
        else:
            model = get_cao_cvfcn_model(input_shape=(None, None, 2 * channels), num_classes=4,
                                        dtype=np.float32, name="cao_rvfcn", dropout_dict=dropout)
    else:
        if complex_mode:
            raise ValueError("Tensorflow does not support complex model. "
                             "Do not use tensorflow and complex_mode both as True")
        model = get_tf_real_cao_model(input_shape=(None, None, 2 * channels), num_classes=4,
                                      name="tf_cao_rvfcn", dropout_dict=dropout)
    model.load_weights(str(root_path / "checkpoints/cp.ckpt"))
    # set_trace()
    return model


def save_result_image_from_saved_model(root_path, complex_mode=True, tensorflow=False, dropout=None, coherency=False,
                                       use_mask=False):
    # Prepare image
    mat, seg = open_data()
    if not coherency:
        full_img = get_k_vector(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    else:
        full_img = get_coherency_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    if not complex_mode:
        full_img, seg = transform_to_real_map_function(full_img, seg)
    full_img = tf.pad(full_img, [[0, 3], [0, 0], [0, 0]])
    full_image = tf.expand_dims(full_img, axis=0)

    # Get model
    model = open_saved_model(root_path,
                             complex_mode=complex_mode, tensorflow=tensorflow, dropout=dropout, coherency=coherency)
    set_trace()
    tf.print("Predicting image")
    prediction = model.predict(full_image)[0]
    tf.print("Prediction done")
    prediction = (tf.math.real(prediction) + tf.math.imag(prediction)) / 2.
    labels_to_ground_truth(prediction, savefig=str(root_path / "prediction"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs=1, type=str,
                        default='/home/barrachina/Documents/onera/PolSar/Bretigny-ONERA/log/2021/09September/24Friday/run-13h46m02',
                        help='Path with the model checkpoint')
    parser.add_argument('--use_mask', action='store_true', help='Set non-labeled pixels to black')
    root_path = Path(parser.parse_args().path[0])
    params = read_parameters(root_path)
    save_result_image_from_saved_model(root_path, complex_mode=params['complex'], tensorflow=params['tensorflow'],
                                       dropout=params['dropout'], coherency=params['coherency'],
                                       use_mask=parser.parse_args().use_mask)
