import scipy.io
import os
import numpy as np
import tensorflow as tf
from pdb import set_trace
from sklearn.model_selection import train_test_split

COLORS = [
    [1, 0.349, 0.392],      # Red; Built-up Area
    [0.086, 0.858, 0.576],  # Green; Wood Land
    [0.937, 0.917, 0.352],   # Yellow; Open Area
    [0, 0.486, 0.745]
]


def mean_filter(input, filter_size=3):
    """
    Performs mean filter on an image with a filter of size (filter_size, filter_size)
    :param input: Image of shape (Height, Width, Channels)
    :param filter_size:
    :return:
    """
    input_transposed = tf.transpose(input, perm=[2, 0, 1])  # Get channels to the start

    filter = tf.ones(shape=(filter_size, filter_size, 1, 1), dtype=input.dtype.real_dtype) / (filter_size * filter_size)
    filtered_T_real = tf.nn.convolution(input=tf.expand_dims(tf.math.real(input_transposed), axis=-1),
                                        filters=filter, padding="SAME")
    filtered_T_imag = tf.nn.convolution(input=tf.expand_dims(tf.math.imag(input_transposed), axis=-1),
                                        filters=filter, padding="SAME")
    filtered_T = tf.complex(filtered_T_real, filtered_T_imag)
    return tf.transpose(tf.squeeze(filtered_T), perm=[1, 2, 0])  # Get channels to the end again


def get_coherency_matrix(HH, VV, HV, kernel_shape=3):
    # Section 2: https://earth.esa.int/documents/653194/656796/LN_Advanced_Concepts.pdf
    k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
    tf_k = tf.expand_dims(tf.transpose(k, perm=[1, 2, 0]), axis=-1)  # From shape 3xhxw to hxwx3x1

    T = tf.linalg.matmul(tf_k, tf.transpose(tf_k, perm=[0, 1, 3, 2], conjugate=True))  # T = k * k^H
    one_channel_T = tf.reshape(T, shape=(T.shape[0], T.shape[1], T.shape[2] * T.shape[3]))  # hxwx3x3 to hxwx9
    filtered_T = mean_filter(one_channel_T, kernel_shape)
    # filtered_T = tfa.image.mean_filter2d(tf.math.real(one_channel_T), kernel_shape)     # Filter the image
    flatten_T = tf.reshape(filtered_T, shape=(filtered_T.shape[0] * filtered_T.shape[1], filtered_T.shape[2]))
    return flatten_T


def remove_unlabeled(x, y):
    """
    Removes the unlabeled pixels from both image and labels
    :param x: image input
    :param y: labels inputs. all values of 0's will be eliminated and its corresponging values of x
    :return: tuple (x, y) without the unlabeled pixels.
    """
    mask = y != 0
    return x[mask], y[mask]


def sparse_to_categorical_1D(labels):
    cat_labels = np.zeros((labels.shape[0], np.max(labels)))
    for i, val in enumerate(labels):
        if val != 0:
            cat_labels[i][val - 1] = 1.
    return cat_labels


def use_neighbors(im, size: int = 3, stride: int = 1, pad: int = 0):
    """
    Extracts many sub-images from one big image.
    :param im: Image dataset of shape (H, W, channels)
    :param size: Size of the desired mini new images
    :param stride: stride between images, use stride=size for images not to overlap
    :param pad: Pad borders
    :return: tuple of numpy arrays (tiles, label_tiles)
    """
    assert im.shape[0] > size and im.shape[1] > size
    output_shape = (
        int(np.floor((im.shape[0] + 2 * pad - size) / stride) + 1),
        int(np.floor((im.shape[1] + 2 * pad - size) / stride) + 1),
        size * size * im.shape[2]
    )
    tiles = np.zeros(shape=output_shape, dtype=im.dtype.as_numpy_dtype)
    if pad:
        im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)))
    for x in range(0, output_shape[0]):
        for y in range(0, output_shape[1]):
            slice_x = slice(x * stride, x * stride + size)
            slice_y = slice(y * stride, y * stride + size)
            tiles[x, y] = np.reshape(im[slice_x, slice_y], -1)
    return tiles


def open_data():
    if os.path.exists('/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'):
        path = '/media/barrachina/data/datasets/PolSar/Bretigny-ONERA/data'
    elif os.path.exists('/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'):
        path = '/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Bretigny-ONERA/data'
    else:
        raise FileNotFoundError("Dataset path not found")
    mat = scipy.io.loadmat(path + '/bretigny_seg.mat')
    seg = scipy.io.loadmat(path + '/bretigny_seg_4ROI.mat')
    return mat, seg


def get_k_vector(HH, VV, HV):
    k = np.array([HH + VV, HH - VV, 2 * HV]) / np.sqrt(2)
    data = use_neighbors(tf.transpose(k, perm=[1, 2, 0]), pad=1)
    return tf.reshape(data, shape=(data.shape[0] * data.shape[1], data.shape[2]))


def format_data_for_mlp(data, labels):
    labels = np.reshape(labels, -1)  # Flatten labels
    assert data.shape[0] == labels.shape[0]
    # Remove unlabeled data
    T, labels = remove_unlabeled(data, labels)
    assert T.shape[0] == labels.shape[0]

    # Split train and test
    x_train, x_test, y_train, y_test = train_test_split(T.numpy(), labels, train_size=0.1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

    # Sparse into categorical labels
    y_test = sparse_to_categorical_1D(y_test)
    y_train = sparse_to_categorical_1D(y_train)
    y_val = sparse_to_categorical_1D(y_val)
    # assert T.shape[1] == 9
    assert len(T.shape) == 2

    return x_train, y_train, x_val, y_val


def get_k_data():
    mat, seg = open_data()
    k = get_k_vector(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    return format_data_for_mlp(k, seg['image'])


def get_coh_data():
    mat, seg = open_data()

    T = get_coherency_matrix(HH=mat['HH'], VV=mat['VV'], HV=mat['HV'])
    return format_data_for_mlp(T, seg['image'])


def to_rgb_colors(labels):
    assert len(labels.shape) == 2
    output = np.zeros(shape=labels.shape + (4,))
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            if labels[i, j] == 0:
                output[i][j] = [0, 0, 0, 0]
            elif labels[i][j] == 1:     # Forest
                output[i][j] = COLORS[1] + [0.8]
            elif labels[i][j] == 2:     # Piste
                output[i][j] = COLORS[3] + [0.8]
            elif labels[i][j] == 3:     # Piste
                output[i][j] = COLORS[0] + [0.8]
            elif labels[i][j] == 4:     # Forest
                output[i][j] = COLORS[2] + [0.8]
    return (output * 255).astype(np.uint8)


def print_image_with_labels():
    try:
        from PIL import Image
    except ImportError:
        import Image

    path = "img3.png"
    background = Image.open(path)
    _, labels = open_data()
    labels = labels['image']
    labels = to_rgb_colors(labels)

    overlay = Image.fromarray(labels, mode="RGBA")
    background = background.convert("RGBA")

    background.paste(overlay.convert('RGB'), (0, 0), overlay)
    background.save("overlapped.png", "PNG")
    # background.show()


if __name__ == "__main__":
    print_image_with_labels()
    # get_k_data()
    # x_train, y_train, x_val, y_val = get_coh_data()
