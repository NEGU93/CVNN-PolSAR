import sys
import tensorflow as tf
from pdb import set_trace
sys.path.insert(1, '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen')
from oberpfaffenhofen_dataset import get_dataset_for_segmentation
from oberpfaffenhofen_unet import get_cao_cvfcn_model

cao_fit_parameters = {
    'validation_split': 0.1,    # Section 3.3.2
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


if __name__ == "__main__":
    dataset = get_dataset_for_segmentation(size=cao_dataset_parameters['sliding_window_size'],
                                           stride=cao_dataset_parameters['sliding_window_stride']).batch(
        cao_dataset_parameters['batch_size']).map(flip)
    # data, label = next(iter(dataset))
    # set_trace()
    model = get_cao_cvfcn_model(input_shape=(cao_dataset_parameters['sliding_window_size'],
                                             cao_dataset_parameters['sliding_window_size'], 21))
    set_trace()
    model.fit(x=dataset, epochs=cao_fit_parameters['epochs'],
              validation_split=cao_fit_parameters['validation_split'], shuffle=True)

