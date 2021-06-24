import sys
import os
from time import time, strftime, localtime
from datetime import timedelta
import tensorflow as tf
from notify_run import Notify
import traceback
from os import path
import numpy as np
from pdb import set_trace
if path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
    NOTIFY = False
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
    NOTIFY = True
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from oberpfaffenhofen_dataset import get_ober_dataset_for_segmentation
from cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model
from cvnn.utils import create_folder

cao_fit_parameters = {
    'epochs': 200                   # Section 3.3.2
}


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def get_checkpoints_list():
    temp_path = create_folder("./log/")
    os.makedirs(temp_path, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)
    return [tensorboard_callback, cp_callback]


def run_model(complex_mode=True, tensorflow=False):
    train_dataset, test_dataset = get_ober_dataset_for_segmentation(complex_mode=complex_mode)
    # data, label = next(iter(dataset))
    if not tensorflow:
        if complex_mode:
            model = get_cao_cvfcn_model(input_shape=(128, 128, 21))
        else:
            model = get_cao_cvfcn_model(input_shape=(128, 128, 42),
                                        dtype=np.float32)
    else:
        if complex_mode:
            raise ValueError("Tensorflow does not support complex model. "
                             "Do not use tensorflow and complex_mode both as True")
        model = get_tf_real_cao_model(input_shape=(128, 128, 42))
    # Checkpoints
    callbacks = get_checkpoints_list()

    start = time()
    history = model.fit(x=train_dataset, epochs=cao_fit_parameters['epochs'],
                        validation_data=test_dataset, shuffle=True, callbacks=callbacks)
    stop = time()
    return secondsToStr(stop - start)


def open_saved_models(checkpoint_path):
    train_dataset, test_dataset = get_ober_dataset_for_segmentation(size=cao_dataset_parameters['sliding_window_size'],
                                                                    stride=cao_dataset_parameters['sliding_window_stride'])
    test_dataset = test_dataset.batch(cao_dataset_parameters['batch_size'])
    model = get_cao_cvfcn_model(input_shape=(cao_dataset_parameters['sliding_window_size'],
                                             cao_dataset_parameters['sliding_window_size'], 21))
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    model.load_weights(checkpoint_path)
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def train_model():
    # https://notify.run/c/PGqsOzNQ1cSGdWM7
    if NOTIFY:
        notify = Notify()
        notify.send('New simulation started')
    try:
        time = run_model(complex_mode=False, tensorflow=True)
        if NOTIFY:
            notify.send(f"Simulations done in {time}")
    except Exception as e:
        if NOTIFY:
            notify.send("Error occurred")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    # run_model()
    train_model()
    # open_saved_models("/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/u-net/log/2021/05May/12Wednesday/run-19h55m20/checkpoints/cp.ckpt")
