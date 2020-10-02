import scipy.io
import numpy as np
from pdb import set_trace


def data_for_test(train_data, T_L2, nwin):
    T11L = T_L2['T11L']
    T12L = T_L2['T12L']
    T13L = T_L2['T13L']
    T22L = T_L2['T22L']
    T23L = T_L2['T23L']
    T33L = T_L2['T33L']

    t11 = train_data[:, :, 0, :]
    t12 = train_data[:, :, 1, :]
    t13 = train_data[:, :, 2, :]
    t22 = train_data[:, :, 3, :]
    t23 = train_data[:, :, 4, :]
    t33 = train_data[:, :, 5, :]
    m, n, q = t11.shape

    T11_ave = sum(sum(sum(t11))) / (m * n * q)
    T12_ave = sum(sum(sum(t12))) / (m * n * q)
    T13_ave = sum(sum(sum(t13))) / (m * n * q)
    T22_ave = sum(sum(sum(t22))) / (m * n * q)
    T23_ave = sum(sum(sum(t23))) / (m * n * q)
    T33_ave = sum(sum(sum(t33))) / (m * n * q)

    T11_std = np.sqrt((sum(sum(sum((t11 - T11_ave) ** 2)))) / (m * n * q))
    T12_std = np.sqrt((sum(sum(sum((t12 - T12_ave) * np.conj(t12 - T12_ave))))) / (m * n * q))
    T13_std = np.sqrt((sum(sum(sum((t13 - T13_ave) * np.conj(t13 - T13_ave))))) / (m * n * q))
    T22_std = np.sqrt((sum(sum(sum((t22 - T22_ave) ** 2)))) / (m * n * q))
    T23_std = np.sqrt((sum(sum(sum((t23 - T23_ave) * np.conj(t23 - T23_ave))))) / (m * n * q))
    T33_std = np.sqrt((sum(sum(sum((t33 - T33_ave) ** 2)))) / (m * n * q))

    TT11 = (t11 - T11_ave) / T11_std
    TT12 = (t12 - T12_ave) / T12_std
    TT13 = (t13 - T13_ave) / T13_std
    TT22 = (t22 - T22_ave) / T22_std
    TT23 = (t23 - T23_ave) / T23_std
    TT33 = (t33 - T33_ave) / T33_std

    train_data_s = np.empty(train_data.shape, dtype=np.complex128)
    train_data_s[:, :, 0, :] = TT11
    train_data_s[:, :, 1, :] = TT12
    train_data_s[:, :, 2, :] = TT13
    train_data_s[:, :, 3, :] = TT22
    train_data_s[:, :, 4, :] = TT23
    train_data_s[:, :, 5, :] = TT33

    T11_all = (T11L - T11_ave) / T11_std
    T12_all = (T12L - T12_ave) / T12_std
    T13_all = (T13L - T13_ave) / T13_std
    T22_all = (T22L - T22_ave) / T22_std
    T23_all = (T23L - T23_ave) / T23_std
    T33_all = (T33L - T33_ave) / T33_std

    row, col = T11L.shape

    K = len(np.arange(0, row-nwin, 3))*len(np.arange(0, col - nwin, 3))
    test_img_Flevoland = np.empty((nwin, nwin, 6, K), dtype=np.complex128)
    k = 0
    for i in np.arange(0, row-nwin, 3):
        for j in np.arange(0, col - nwin, 3):
            test_img_Flevoland[:, :, 0, k] = T11_all[i:i+nwin, j:j+nwin]
            test_img_Flevoland[:, :, 1, k] = T12_all[i:i+nwin, j:j+nwin]
            test_img_Flevoland[:, :, 2, k] = T13_all[i:i+nwin, j:j+nwin]
            test_img_Flevoland[:, :, 3, k] = T22_all[i:i+nwin, j:j+nwin]
            test_img_Flevoland[:, :, 4, k] = T23_all[i:i+nwin, j:j+nwin]
            test_img_Flevoland[:, :, 5, k] = T33_all[i:i+nwin, j:j+nwin]
            k += 1

    # Test my function
    # test_img_Flevoland_mat = scipy.io.loadmat('/home/barrachina/Document
    # s/GitHub/CV-CNN/Test Demo/test_img_Flavoland.mat')['test_img_Flevoland']
    # I cast to allow small rounding differences
    # assert (test_img_Flevoland.astype(np.complex64) == test_img_Flevoland_mat.astype(np.complex64)).all()
    return test_img_Flevoland, train_data_s


def get_trainings_labels():
    set_trace()


if __name__ == "__main__":
    train_data = scipy.io.loadmat('/home/barrachina/Documents/GitHub/CV-CNN/Test Demo/train_data.mat')['train_data']
    T_L2 = scipy.io.loadmat('/home/barrachina/Documents/GitHub/CV-CNN/Test Demo/T_L2.mat')
    test_img_Flevoland, train_data_s = data_for_test(train_data, T_L2, nwin=12)
    label = scipy.io.loadmat('/home/barrachina/Documents/GitHub/CV-CNN/Test Demo/label.mat')['label']
    set_trace()
