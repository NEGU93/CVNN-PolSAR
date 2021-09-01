import sys
import os
import pickle
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
elif path.exists('W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar'):
    sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\datasets\PolSar')
    NOTIFY = False
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from oberpfaffenhofen_dataset import get_ober_dataset_for_segmentation
from cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model, get_debug_tf_models
from cvnn.utils import create_folder
from cvnn.montecarlo import MonteCarlo
from tensorflow.keras.utils import plot_model

cao_fit_parameters = {
    'epochs': 200              # Section 3.3.2
}


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def get_callbacks_list():
    temp_path = create_folder("./log/")
    os.makedirs(temp_path, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=temp_path / 'checkpoints/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=0, save_best_only=True)
    return [tensorboard_callback, cp_callback], temp_path


def run_model(complex_mode=True, tensorflow=False):
    train_dataset, test_dataset = get_ober_dataset_for_segmentation(complex_mode=complex_mode)
    # data, label = next(iter(dataset))
    if not tensorflow:
        if complex_mode:
            model = get_cao_cvfcn_model(input_shape=(None, None, 21))
        else:
            model = get_cao_cvfcn_model(input_shape=(None, None, 42), dtype=np.float32)
    else:
        if complex_mode:
            raise ValueError("Tensorflow does not support complex model. "
                             "Do not use tensorflow and complex_mode both as True")
        model = get_tf_real_cao_model(input_shape=(None, None, 42))
    # Checkpoints
    callbacks, temp_path = get_callbacks_list()
    # elem, label = next(iter(test_dataset))
    # input_out = model.layers[0](elem)
    start = time()
    history = model.fit(x=train_dataset, epochs=cao_fit_parameters['epochs'],
                        validation_data=test_dataset, shuffle=True, callbacks=callbacks)
    stop = time()
    with open(temp_path / 'history_dict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return secondsToStr(stop - start)


def open_saved_models(checkpoint_path):
    train_dataset, test_dataset = get_ober_dataset_for_segmentation()
    test_dataset = test_dataset.batch(100)
    model = get_cao_cvfcn_model(input_shape=(None, None, 21))
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
        time = run_model(complex_mode=True, tensorflow=False)
        if NOTIFY:
            notify.send(f"CV-FCNN Simulations done in {time}")
        time = run_model(complex_mode=True, tensorflow=False)
        if NOTIFY:
            notify.send(f"RV-FCNN Simulations done in {time}")
        time = run_model(complex_mode=False, tensorflow=True)
        if NOTIFY:
            notify.send(f"RV-tf-FCNN Simulations done in {time}")
    except Exception as e:
        if NOTIFY:
            notify.send("Error occurred")
        print(e)
        traceback.print_exc()


def debug_models(indx):
    notify = Notify()
    model = get_debug_tf_models(input_shape=(None, None, 42), indx=indx)
    train_dataset, test_dataset = get_ober_dataset_for_segmentation(complex_mode=False)
    # for model in models_list:
    notify.send(f"Testing model {indx}: {model.name}")
    try:
        callbacks, temp_path = get_callbacks_list()
        # plot_model(model, to_file=temp_path / "model.png", show_shapes=True)
        history = model.fit(x=train_dataset, epochs=200, validation_data=test_dataset, shuffle=True, callbacks=callbacks)
        with open(temp_path / 'history_dict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    except Exception as e:
        notify.send("Error occurred")
        print(e)
        traceback.print_exc()


def run_montecarlo():
    train_dataset, test_dataset = get_ober_dataset_for_segmentation(complex_mode=False)
    montecarlo = MonteCarlo()
    montecarlo.add_model(get_cao_cvfcn_model(input_shape=(None, None, 42), dtype=np.float32))
    montecarlo.add_model(get_tf_real_cao_model(input_shape=(None, None, 42)))
    montecarlo.run(x=train_dataset, y=None, data_summary="Oberpfaffenhofen_PolInSAR",
                   validation_data=test_dataset, validation_split=0.0,
                   iterations=5, epochs=200, shuffle=True)


if __name__ == "__main__":
    # run_model(complex_mode=False, tensorflow=True)
    # train_model()
    # args = sys.argv
    # debug_models(int(args[1]))
    run_montecarlo()
    # open_saved_models("/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen/u-net/log/2021/05May/12Wednesday/run-19h55m20/checkpoints/cp.ckpt")
