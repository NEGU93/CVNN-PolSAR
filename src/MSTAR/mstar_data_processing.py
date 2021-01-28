from pathlib import Path
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.model_selection import train_test_split

root_path = Path('/media/barrachina/data/datasets/MSTAR/')

df = pd.read_pickle(root_path / 'data.pkl')

classes = df.TargetType.unique()

sparse_labels = {}
for i, cla in enumerate(classes):
    sparse_labels[cla] = i


def get_data():
    images = []
    labels = []
    for i, row in df.iterrows():
        images.append(row['data'])
        labels.append(sparse_labels[row['TargetType']])
    return images, labels


def resize_images(images, labels):
    cropped_images = []
    cropped_labels = []
    for i, img in enumerate(images):
        if img.shape[0] >= 128 and img.shape[1] >= 128:
            cropped_images.append(crop_center(img, 128, 128))
            cropped_labels.append(labels[i])
    return cropped_images, cropped_labels


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def get_train_and_test(train_ratio=0.8):
    images, labels = get_data()
    images, labels = resize_images(images, labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=train_ratio, shuffle=True)
    for i in range(len(classes)):
        print(f"Class {classes[i]} ({i}) is present {y_train.count(i)} times in train and {y_test.count(i)} in test.")
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
