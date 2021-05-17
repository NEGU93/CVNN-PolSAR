from urban_sound_load_dataset import mel_spectrum
from cvnn.dataset import Dataset
from cvnn.montecarlo import mlp_run_real_comparison_montecarlo
from pdb import set_trace
import numpy as np
import librosa


def get_fold(k):
    mel_spec, label = mel_spectrum(debug=False)
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    for i in range(10):
        phase = np.angle(mel_spec[i])
        amplitude = np.abs(mel_spec[i])
        amplitude = amplitude / amplitude.max()     # Normalize
        if not np.all(amplitude <= 1):      #, "Correctly normalized"
            set_trace()
        log_amplitude = librosa.power_to_db(amplitude)
        norm_log_amplitude = (log_amplitude - log_amplitude.min()) / (log_amplitude.max() - log_amplitude.min())
        log_mel_spec = norm_log_amplitude * np.exp(1j*phase)
        if i == k:
            x_test = log_mel_spec
            y_test = label[i]
        else:
            if x_train.size == 0:
                x_train = log_mel_spec
                y_train = label[i]
            else:
                x_train = np.concatenate((x_train, log_mel_spec), axis=0)
                y_train = np.concatenate((y_train, label[i]), axis=0)
    x_train = np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
    y_test = Dataset.sparse_into_categorical(y_test)
    y_train = Dataset.sparse_into_categorical(y_train)
    train_dataset = Dataset(x_train, y_train, dataset_name="Log-mel-spectrogram")
    return train_dataset, (x_test, y_test)


def train_10_fold_montecarlo(folds=10, iterations=10, shape_raw=None, polar=True, batch_size=100, dropout=0.5):
    paths = []
    assert folds <= 10
    if shape_raw is None:
        shape_raw = [2000, 1000]
    for k in range(folds):  # K-fold validation
        # 1. PREPARE DATASET
        train_dataset, validation_dataset = get_fold(k)
        # 2. TRAIN
        path = mlp_run_real_comparison_montecarlo(dataset=train_dataset, do_conf_mat=False, batch_size=batch_size,
                                                  validation_data=validation_dataset, validation_split=0.0,
                                                  iterations=iterations, polar=polar, shape_raw=shape_raw,
                                                  dropout=dropout)
        paths.append(path)


if __name__ == '__main__':
    train_10_fold_montecarlo(folds=1, shape_raw=[2000, 2000], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[2000, 2000], dropout=0.8)
    train_10_fold_montecarlo(folds=1, shape_raw=[2000, 2000], dropout=0.7)
    train_10_fold_montecarlo(folds=1, shape_raw=[2000, 2000], dropout=0.6)

    train_10_fold_montecarlo(folds=1, shape_raw=[3000], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[3000], dropout=0.8)
    train_10_fold_montecarlo(folds=1, shape_raw=[3000], dropout=0.7)
    train_10_fold_montecarlo(folds=1, shape_raw=[3000], dropout=0.6)

    train_10_fold_montecarlo(folds=1, shape_raw=[200, 200], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[200, 200], dropout=0.5)

    train_10_fold_montecarlo(folds=1, shape_raw=[300], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[300], dropout=0.5)

    train_10_fold_montecarlo(folds=1, shape_raw=[1500], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[1500], dropout=0.7)
    train_10_fold_montecarlo(folds=1, shape_raw=[1500], dropout=0.5)

    train_10_fold_montecarlo(folds=1, shape_raw=[1000, 1000], dropout=0.9)
    train_10_fold_montecarlo(folds=1, shape_raw=[1000, 1000], dropout=0.7)
    train_10_fold_montecarlo(folds=1, shape_raw=[1000, 1000], dropout=0.5)

    train_10_fold_montecarlo(folds=1, shape_raw=[1000], dropout=0.5)
    train_10_fold_montecarlo(folds=1, shape_raw=[500, 500], dropout=0.5)
