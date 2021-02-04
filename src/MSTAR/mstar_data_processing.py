from pathlib import Path
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from pdb import set_trace
from sklearn.model_selection import train_test_split

<<<<<<< HEAD
root_path = Path('/home/cfren/Bureau/Documents/onera/MSTARdata')
=======
DEBUG = False
root_path = Path('/media/barrachina/data/datasets/MSTAR/Targets')
>>>>>>> 5fc137dfe3ca242b9d3a23aa022c30c321cfca48

df = pd.read_pickle(root_path / 'data.pkl')

classes = df.TargetType.unique()
classes = np.delete(classes, np.argwhere(classes == 'slicey'))

sparse_labels = {}
for i, cla in enumerate(classes):
    sparse_labels[cla] = i


def get_data():
    images = []
    labels = []
    for _, row in df.iterrows():
        images.append(row['data'])
        labels.append(sparse_labels[row['TargetType']])
    return images, labels


def resize_image(img, input_shape=(128, 128)):
    """
    Resizes images to `input_shape` size.
        If image is smaller it zero pads.
        If image is bigger it crops.
        If image has one axis bigger and one smaller it will both zero pad the smaller axis and crop the biggest one.
    ATTENTION (TODO?):
        If the desired shape is an even size (ex. 128x128) and the image shape is smaller, t
        the image shape should also be even (ex. 58x58).
    :param img: Image to be resized.
    :param input_shape: Desired shape
    :return: reshaped image of shape `input_shape`
    """
    assert img.shape[0] % 2 == input_shape[0] % 2 or img.shape[0] >= input_shape[0]
    assert img.shape[1] % 2 == input_shape[1] % 2 or img.shape[1] >= input_shape[1]
    if img.shape[0] < input_shape[0] or img.shape[1] < input_shape[1]:
        img = np.pad(img,
                     (max(0, int((input_shape[0] - img.shape[0])/2)), max(0, int((input_shape[1] - img.shape[1])/2))),
                     mode='constant')
    if img.shape[0] > input_shape[0] or img.shape[1] > input_shape[1]:
        img = crop_center(img, input_shape[0], input_shape[1])
    return img


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def separate_by_angle():
    """
    Gets a dataset dictionary of the form:
        {
            '15': [{}, ..., {}]
            '17': [
                    {'image': np.array(, dtype=np.complex64), 'label': 'Target_Type'},
                    ..., {}
                ]
            '30': [{}, ..., {}]
        }
    :return:
    """
    dataset_by_angle = {}        # Dictionary with key = Desired Depression (15, 17, 30)
    for _, row in df.iterrows():
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        if (depression := dataset_by_angle.get(row['DesiredDepression'])) is None:
            dataset_by_angle[row['DesiredDepression']] = depression = []
        if row['TargetType'] == 'slicey':
            continue
        if row['TargetType'] == 'bmp2_tank' and not row['path'].endswith('SN_C21'):
            continue    # Only use snc21 of the bmp2 tanks
        if row['TargetType'] == 't72_tank' and not row['path'].endswith('SN_132'):
            continue    # Only use sn132 of the t72 tanks
        depression.append({'image': row['data'], 'label': row['TargetType']})
    if DEBUG:
        print(f"Data angles: {dataset_by_angle.keys()}")
    return dataset_by_angle


def separate_train_and_test_with_angle(dataset, train_angle: int = 17, test_angle: int = 15):
    """
    Gets train and test set based on a desired angle.
    :param dataset: Dataset to be divided.
    :param train_angle:
    :param test_angle:
    :return:
    """
    train_set = dataset[train_angle]
    test_set = dataset[test_angle]
    train_shapes = set()
    test_shapes = set()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for element in train_set:
        img = resize_image(element['image'])
        train_shapes.add(img.shape)
        x_train.append(img)
        y_train.append(sparse_labels[element['label']])
    for element in test_set:
        img = resize_image(element['image'])
        test_shapes.add(img.shape)
        x_test.append(img)
        y_test.append(sparse_labels[element['label']])
    # First remark, shapes are not all 128x128 as the paper says.
    if DEBUG:
        print(f"train_shapes: {train_shapes}")
        print(f"test_shapes: {test_shapes}")
    return x_train, x_test, y_train, y_test


def get_train_and_test():
    dataset = separate_by_angle()
    x_train, x_test, y_train, y_test = separate_train_and_test_with_angle(dataset=dataset)
    if DEBUG:
        t = PrettyTable(['Class', 'Train total', 'Test total'])
        for i in range(len(classes)):
            t.add_row([classes[i], y_train.count(i), y_test.count(i)])
        print(t)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    DEBUG = True
    get_train_and_test()
