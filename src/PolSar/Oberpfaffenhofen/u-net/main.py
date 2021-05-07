import sys
import os
from time import time, strftime, localtime
from datetime import timedelta
import tensorflow as tf
from pdb import set_trace
sys.path.insert(1, '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen')
from oberpfaffenhofen_dataset import get_dataset_for_segmentation
from oberpfaffenhofen_unet import get_cao_cvfcn_model
from cvnn.utils import create_folder

cao_fit_parameters = {
    'epochs': 200               # Section 3.3.2
}
cao_dataset_parameters = {
    'batch_size': 30,               # Section 3.3.2
    'sliding_window_size': 128,     # Section 3.3.2
    'sliding_window_stride': 25     # Section 3.3.2
}


def flip(data, labels):
    """
    Flip augmentation
    :param data: Image to flip
    :param labels: Image labels
    :return: Augmented image
    """
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)

    return data, labels


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset_for_segmentation(size=cao_dataset_parameters['sliding_window_size'],
                                                               stride=cao_dataset_parameters['sliding_window_stride'])
    train_dataset = train_dataset.batch(cao_dataset_parameters['batch_size']).map(flip)
    test_dataset = test_dataset.batch(cao_dataset_parameters['batch_size'])
    # data, label = next(iter(dataset))
    # set_trace()
    model = get_cao_cvfcn_model(input_shape=(cao_dataset_parameters['sliding_window_size'],
                                             cao_dataset_parameters['sliding_window_size'], 21))
    temp_path = create_folder("./log/")
    os.makedirs(temp_path, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=temp_path / 'tensorboard', histogram_freq=0)
    callbacks = [tensorboard_callback]

    start = start_time = time()
    history = model.fit(x=train_dataset, epochs=2,   # cao_fit_parameters['epochs'],
                        validation_data=test_dataset, shuffle=True, callbacks=callbacks)
    stop = time()
    print(f'Time: {secondsToStr(stop - start)}')


