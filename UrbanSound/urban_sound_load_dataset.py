import pandas as pd
from pathlib import Path
from pdb import set_trace
import librosa
from tqdm import tqdm
import librosa.display
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import pydub
import csv
import os

PATH = Path('/media/barrachina/data/datasets/UrbanSound8k/UrbanSound8K')


def duration_stats():
    labels = [[] for _ in range(10)]
    duration_lst = [[] for _ in range(10)]
    path = Path('/media/barrachina/data/datasets/UrbanSound8k/UrbanSound8K')
    metadata = pd.read_csv(path / 'metadata/UrbanSound8K.csv')
    for _, row in metadata.iterrows():
        full_path = path / 'audio' / ('fold' + str(row['fold'])) / str(row['slice_file_name'])
        fold_idx = row['fold'] - 1
        labels[fold_idx].append(row['classID'])
        duration_lst[fold_idx].append(librosa.get_duration(filename=full_path))
    file = open(path / "metadata/duration/4sec_occurrences.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(['fold', 'occurrences', 'class0', 'class1', 'class2', 'class3', 'class4',
                     'class5', 'class6', 'class7', 'class8', 'class9'])
    for i in range(len(duration_lst)):
        np.save(path / ("metadata/duration/" + 'fold' + str(i + 1) + ".npy"), np.array(duration_lst[i]))
        plt.hist(duration_lst[i], bins=range(0, 5))
        plt.xlabel('time (seconds)')
        plt.savefig(path / ("metadata/duration/" + 'fold' + str(i + 1) + ".svg"))
        plt.clf()
        to_write = [i + 1, duration_lst[i].count(4.0)]
        indices = [x == 4.0 for x in duration_lst[i]]
        labels[i] = list(compress(labels[i], indices))
        for cls in range(10):
            to_write.append(labels[i].count(cls))
        writer.writerow(to_write)
    file.close()


def do_stats():
    labels = [[] for _ in range(10)]
    metadata = pd.read_csv(PATH / 'metadata/UrbanSound8K.csv')
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        fold_idx = row['fold'] - 1
        labels[fold_idx].append(row['classID'])
    print("Saving stats...")
    with open(PATH / "metadata/class_occurrences.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['fold', 'total occurrences', 'class0', 'class1', 'class2', 'class3', 'class4',
                         'class5', 'class6', 'class7', 'class8', 'class9'])
        for fld in range(len(labels)):
            to_write = [fld + 1, len(labels[fld])]
            for cls in range(10):
                to_write.append(labels[fld].count(cls))
            writer.writerow(to_write)


def _get_path(duration_ms=4000, pad=False, filtered=False):
    sub_folder = str("padded" if pad else "cropped") + str(int(np.ceil(duration_ms / 1000))) \
                 + "seconds/" + str("filtered7classes/" if filtered else "")
    os.makedirs(PATH / ("numpy_arrays/" + sub_folder + "raw_audio/"), exist_ok=True)
    os.makedirs(PATH / ("numpy_arrays/" + sub_folder + "labels/"), exist_ok=True)
    return PATH / ("numpy_arrays/" + sub_folder)


def _load_audio_audiosegment(path, duration_ms=4000, pad=True):
    sr = 22050
    audio = pydub.AudioSegment.from_wav(path)
    if pad:
        silence = pydub.AudioSegment.silent(duration=duration_ms)
        audio = silence.overlay(audio)
    samples = audio.set_frame_rate(sr).split_to_mono()[0].get_array_of_samples()
    fp_arr = np.array(samples).astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    if len(fp_arr) == 88199:
        fp_arr = np.append(fp_arr, 0.0)
    if len(fp_arr) != 88200:
        set_trace()
    return fp_arr, sr


def _load_audio_librosa(path, duration_ms: int = 4000, pad: bool = True):
    y, sr = librosa.core.load(path, mono=True, duration=np.ceil(duration_ms/1000))
    if pad:
        silence = np.zeros(int(sr*duration_ms/1000))
        silence[:len(y)] = y
        y = silence
    if len(y) != int(sr*duration_ms/1000):
        return None, None
    return y, sr


def _load_numpy_existing_dataset(path):
    audio = [[] for _ in range(10)]  # 10 folds
    labels = [[] for _ in range(10)]
    number_of_folds = 10
    for fold in range(number_of_folds):
        if os.path.exists(path / ("raw_audio/fold" + str(fold+1) + "_audio.npy")):
            audio[fold] = np.load(path / ("raw_audio/fold" + str(fold+1) + "_audio.npy"))
        else:
            print("Fold " + str(fold) + " audio file not found")
        if os.path.exists(path / ("labels/fold" + str(fold+1) + "_labels.npy")):
            labels[fold] = np.load(path / ("labels/fold" + str(fold+1) + "_labels.npy"))
        else:
            print("Fold " + str(fold) + " label file not found")
    return audio, labels


def _load_wav_dataset(path, duration_ms=4000, pad=False):
    audio = [[] for _ in range(10)]     # 10 folds
    labels = [[] for _ in range(10)]

    metadata = pd.read_csv(PATH / 'metadata/UrbanSound8K.csv')
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        full_path = PATH / 'audio' / ('fold' + str(row['fold'])) / str(row['slice_file_name'])
        fold_idx = row['fold'] - 1
        raw_audio, sr = _load_audio_librosa(full_path, duration_ms=duration_ms, pad=pad)
        if raw_audio is not None:
            labels[fold_idx].append(row['classID'])
            audio[fold_idx].append(raw_audio)
    for i in range(len(labels)):
        np.save(path / ("labels/" + 'fold' + str(i + 1) + "_labels.npy"), np.array(labels[i]))
        np.save(path / ("raw_audio/" + 'fold' + str(i + 1) + "_audio.npy"), np.array(audio[i]))
    return audio, labels


def load_dataset(duration_ms: int = 4000, pad: bool = False):
    ret_path = _get_path(duration_ms=duration_ms, pad=pad)
    if any(File.endswith(".npy") for File in os.listdir(ret_path / "raw_audio")):
        # npy files have already been saved
        audio, labels = _load_numpy_existing_dataset(ret_path)
    else:
        # First run
        audio, labels = _load_wav_dataset(ret_path, duration_ms=duration_ms, pad=pad)
    return audio, labels


def filter_dataset(classes=None, duration_ms: int = 4000, pad: bool = False):
    audio, label = load_dataset(duration_ms=duration_ms, pad=pad)
    if classes is None:
        classes = [0, 2, 4, 5, 7, 8, 9]
    path = _get_path(duration_ms=duration_ms, pad=pad, filtered=True)
    os.makedirs(path / "raw_audio", exist_ok=True)
    os.makedirs(path / "labels", exist_ok=True)
    for i in range(len(label)):
        if os.path.exists(path / ("raw_audio/fold" + str(i+1) + "_audio.npy")):
            # print("Loading filtered fold " + str(i+1))
            audio[i] = np.load(path / ("raw_audio/fold" + str(i+1) + "_audio.npy"))
            label[i] = np.load(path / ("labels/fold" + str(i + 1) + "_labels.npy"))
        else:
            # print("Filtering fold " + str(i + 1))
            filter = [lab in classes for lab in label[i]]
            label[i] = np.array(list(compress(label[i], filter)))
            map = 0
            for val in classes:
                label[i][label[i] == val] = map
                map += 1
            audio[i] = np.array(list(compress(audio[i], filter)))
            np.save(path / ("raw_audio/fold" + str(i+1) + "_audio.npy"), audio[i])
            np.save(path / ("labels/fold" + str(i + 1) + "_labels.npy"), label[i])
    return audio, label


def _compute_mel_spectrogram(duration_ms=4000, pad=False, n_fft=2048, hop_length=1024, n_mels=60):
    sr = 22050  # librosa default
    audio, label = filter_dataset(duration_ms=duration_ms, pad=pad)
    mel_spect_complex = [[] for _ in range(10)]
    for fold_index, fold_data in enumerate(audio):
        for stream in fold_data:
            # mel_spect_energy = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=1024)
            fft_windows = librosa.stft(stream, n_fft=n_fft, hop_length=hop_length)  # Returns 2D array [f, t]
            # magnitude = np.abs(fft_windows) ** 2
            mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr / 2.0)
            mel_spect_complex[fold_index].append(mel.dot(fft_windows))
    for i in range(len(mel_spect_complex)):
        path = _get_path(duration_ms=duration_ms, pad=pad, filtered=True)
        os.makedirs(path / "mel_spect", exist_ok=True)
        np.save(path / ("mel_spect/fold" + str(i+1) + "_mel_spec.npy"), mel_spect_complex[i])
    return mel_spect_complex, label


def _load_numpy_existing_mel_spec(path):
    mel_spec = [[] for _ in range(10)]  # 10 folds
    labels = [[] for _ in range(10)]
    number_of_folds = 10
    for fold in range(number_of_folds):
        if os.path.exists(path / ("mel_spect/fold" + str(fold + 1) + "_mel_spec.npy")):
            mel_spec[fold] = np.load(path / ("mel_spect/fold" + str(fold + 1) + "_mel_spec.npy"))
        else:
            print("Fold " + str(fold) + " mel_spect file not found")
        if os.path.exists(path / ("labels/fold" + str(fold + 1) + "_labels.npy")):
            labels[fold] = np.load(path / ("labels/fold" + str(fold + 1) + "_labels.npy"))
        else:
            print("Fold " + str(fold) + " label file not found at "
                  + str(path / ("labels/fold" + str(fold + 1) + "_labels.npy")))
    return mel_spec, labels


def mel_spectrum(duration_ms: int = 4000, pad: bool = False, filtered: bool = True, debug: bool = False):
    path = _get_path(duration_ms=duration_ms, pad=pad, filtered=filtered)
    os.makedirs(path / "mel_spect", exist_ok=True)
    if any(File.endswith(".npy") for File in os.listdir(path / "mel_spect")):
        # npy files have already been saved
        mel_spec, labels = _load_numpy_existing_mel_spec(path)
    else:
        # First run
        mel_spec, labels = _compute_mel_spectrogram(duration_ms=duration_ms, pad=pad)
    if debug:
        librosa.display.specshow(np.abs(mel_spec[0][0]), y_axis='mel', fmax=8000, x_axis='time')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    return mel_spec, labels


if __name__ == '__main__':
    mel_spec, label = mel_spectrum(debug=False)