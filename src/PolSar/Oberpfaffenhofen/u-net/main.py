import sys
import tensorflow as tf
from pdb import set_trace
sys.path.insert(1, '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen')
from oberpfaffenhofen_dataset import get_dataset_for_segmentation
from oberpfaffenhofen_unet import get_model


def flip(data, labels):
    """Flip augmentation
    :param data: Image to flip
    :param labels: Image labels
    :return: Augmented image
    """
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)

    return data, labels


if __name__ == "__main__":
    SIZE = 128
    dataset = get_dataset_for_segmentation(size=SIZE, stride=25).batch(30).map(flip)
    # data, label = next(iter(dataset))
    # set_trace()
    model = get_model(input_shape=(SIZE, SIZE, 21))

