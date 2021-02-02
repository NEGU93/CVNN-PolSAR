from pathlib import Path
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.model_selection import train_test_split

root_path = Path('/media/barrachina/data/datasets/MSTAR/MixedTargets')

df = pd.read_pickle(root_path / 'data.pkl')

classes = df.TargetType.unique()

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


def resize_images(images, labels, input_shape=(128, 128)):
    cropped_images = []
    cropped_labels = []
    for i, img in enumerate(images):
        if img.shape[0] >= input_shape[0] and img.shape[1] >= input_shape[1]:
            cropped_images.append(crop_center(img, input_shape[0], input_shape[1]))
            cropped_labels.append(labels[i])
    return cropped_images, cropped_labels


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
    #
    #
    dataset_by_angle = {}        # Dictionary with key = Desired Depression (15, 17, 30)
    for _, row in df.iterrows():
        if (depression := dataset_by_angle.get(row['DesiredDepression'])) is None:
            dataset_by_angle[row['DesiredDepression']] = depression = []
        depression.append({'image': row['data'], 'label': row['TargetType']})
    print(dataset_by_angle.keys())
    return dataset_by_angle


def separate_train_and_test_with_angle(dataset, train_angle=17, test_angle=15):
    train_set = dataset[train_angle]
    test_set = dataset[test_angle]
    x_train = [element['image'] for element in train_set]
    y_train = [sparse_labels[element['label']] for element in train_set]
    x_test = [element['image'] for element in test_set]
    y_test = [sparse_labels[element['label']] for element in test_set]
    return x_train, x_test, y_train, y_test


def get_train_and_test():
    dataset = separate_by_angle()
    x_train, x_test, y_train, y_test = separate_train_and_test_with_angle(dataset=dataset)
    for i in range(len(classes)):
        print(f"Class {classes[i]} ({i}) is present {y_train.count(i)} times in train and {y_test.count(i)} in test.")
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    get_train_and_test()
