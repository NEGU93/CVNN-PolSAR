import logging
import sys
import os
from collections import defaultdict
import time
from datetime import timedelta
import numpy as np
from pdb import set_trace
from dataset_readers.sf_data_reader import SanFranciscoDataset
from dataset_readers.flevoland_data_reader import FlevolandDataset
from dataset_readers.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from dataset_readers.bretigny_dataset import BretignyDataset
from dataset_readers.garon_dataset import GaronDataset
sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from principal_simulation import DATASET_META


def coh_matrix_generator(dataset_handler, kernel_shape=1):
    dataset_handler.mode = 't'
    coh = dataset_handler.image
    dataset_handler.mode = 's'
    raw_s = dataset_handler.image   # s_11, s_12, s_22
    np_manual_coh = dataset_handler.numpy_coh_matrix(HH=raw_s[:, :, 0], VV=raw_s[:, :, 2], HV=raw_s[:, :, 1],
                                                     kernel_shape=kernel_shape)
    assert np.allclose(coh, np_manual_coh)
    dataset_handler_s = SanFranciscoDataset(mode='s', dataset_name="SF-AIRSAR")
    dataset_handler_t = SanFranciscoDataset(mode='t', dataset_name="SF-AIRSAR")
    dataset_handler_k = SanFranciscoDataset(mode='k', dataset_name="SF-AIRSAR")
    assert np.all(dataset_handler_t.image == dataset_handler_t.get_coherency_matrix(kernel_shape=1))
    assert np.allclose(dataset_handler_t.image, dataset_handler_s.get_coherency_matrix(kernel_shape=1))
    assert np.allclose(dataset_handler_k.get_coherency_matrix(kernel_shape=1),
                       dataset_handler_s.get_coherency_matrix(kernel_shape=1))


def verify_images(handler, gt, img, show_gt, show_img):
    tmp_gt = handler.print_ground_truth(showfig=show_gt, transparent_image=True)
    tmp_img = handler.print_image_png(showfig=show_img)
    if gt is not None:
        assert np.allclose(gt, tmp_gt)
    if img is not None:
        assert np.allclose(tmp_img, img)
    return tmp_img, tmp_gt


def handler_to_test(dataset_handler, show_gt=False, show_img=False):
    ground_truth = None
    rgb_image = None
    for dataset_method in ["separate", "random", "single_separated_image"]:
        np_data = dataset_handler.get_dataset(method=dataset_method, percentage=(0.8, 0.2), size=128, stride=25, pad=0,
                                              shuffle=True, savefig=None, classification=False, cast_to_np=True)
        # verify_labels_balanced(np_data[0][1])
        rgb_image, ground_truth = verify_images(dataset_handler, ground_truth, rgb_image, show_gt, show_img)
    for dataset_method in ["separate", "random"]:
        dataset_handler.get_dataset(method=dataset_method, percentage=(0.8, 0.2), size=128, stride=25, pad=0,
                                    shuffle=True, savefig=None, classification=True, cast_to_np=True)
        rgb_image, ground_truth = verify_images(dataset_handler, ground_truth, rgb_image, show_gt, show_img)


def test_sf():
    dataset_handler = SanFranciscoDataset(mode='s', dataset_name="SF-AIRSAR")
    full_verify_dataset(dataset_handler)


def test_flev():
    dataset_handler = FlevolandDataset(mode='s')
    full_verify_dataset(dataset_handler)


def test_ober():
    dataset_handler = OberpfaffenhofenDataset(mode='s')
    full_verify_dataset(dataset_handler)


def test_bretigny():
    dataset_handler = BretignyDataset(mode='s')
    full_verify_dataset(dataset_handler)


def full_verify_dataset(dataset_handler):
    startt = time.monotonic()
    logging.info(f"Testing dataset {dataset_handler.name}")
    # test_tf_tensor_mode(dataset_handler)
    balance_test_segmentation(dataset_handler)
    handler_to_test(dataset_handler)
    balanced_classification_separate_test(dataset_handler)
    if dataset_handler.name != "FLEVOLAND":
        balanced_classification_random_test(dataset_handler, percentage=(0.04, 0.96))
        balanced_classification_random_test(dataset_handler, percentage=(0.03, 0.02, 0.95),
                                            balance_dataset=[True, True, False])
    balanced_classification_random_test(dataset_handler, percentage=(0.03, 0.02, 0.95),
                                        balance_dataset=False)
    try:
        scattering_vector(dataset_handler)
        mode_change(dataset_handler)
        coh_matrix_generator(dataset_handler)
    except ValueError as e:
        if dataset_handler.name not in ["OBER", "FLEVOLAND"]:       # These datasets only support t mode.
            raise e                                                 # Should catch the error
    logging.info(f"Time testing {dataset_handler.name} was {timedelta(seconds=time.monotonic() - startt)}")


def balanced_classification_random_test(dataset_handler, percentage, balance_dataset=None):
    if balance_dataset is None:
        balance_dataset = [True, False]
    list_ds = dataset_handler.get_dataset(method="random", percentage=percentage, size=6, stride=1, pad=0,
                                          shuffle=True, savefig=None, classification=True,
                                          balance_dataset=balance_dataset, cast_to_np=False)
    if np.any(balance_dataset):
        for i in range(sum(balance_dataset)):
            labels = np.concatenate([x[1] for x in list_ds[i]], axis=0)
            train_sparse = np.argmax(labels, axis=-1)
            train_count = np.bincount(train_sparse)
            assert np.all(train_count[np.nonzero(train_count)] == train_count[np.nonzero(train_count)][0])
    else:
        for i in range(len(percentage)):
            labels = np.concatenate([x[1] for x in list_ds[i]], axis=0)
            total = [lb.shape[0] for lb in labels]
        for i, p in enumerate(percentage):
            assert np.isclose(p, total[i] / sum(total), rtol=0.1)


def balanced_classification_separate_test(dataset_handler):
    list_ds = dataset_handler.get_dataset(method="separate", size=6, stride=1, pad=0,
                                          percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          shuffle=True, savefig=None, classification=True,
                                          cast_to_np=True,
                                          balance_dataset=[True] * len(DATASET_META[dataset_handler.name]["percentage"])
                                          )
    for i in range(len(DATASET_META[dataset_handler.name]["percentage"])):
        train_sparse = np.argmax(list_ds[i][1], axis=-1)
        train_count = np.bincount(train_sparse)
        assert np.all(train_count[np.nonzero(train_count)] == train_count[np.nonzero(train_count)][0])


def garon_balance_test():
    dataset_handler = GaronDataset(mode='s', image_number=1)
    full_verify_dataset(dataset_handler)


def verify_labels_balanced(label_patches):
    """
    Verifies:
        - Total pixels per class balanced too
    Raises assertion error if image is not balanced
    :param label_patches:
    :return:
    """
    # count = np.bincount(np.where(label_patches == 1)[-1])  # Count of total pixels
    # assert np.all(np.logical_or(count == count[np.nonzero(count)][0], count == 0))
    counter = defaultdict(lambda: {"total": 0, "mixed": 0, "solo images": 0, "mixed images": 0})
    for i, la in enumerate(label_patches):
        present_classes = np.where(la == 1)[-1]     # Find all classes (again, there will be at least one).
        assert len(present_classes)                 # No empty image are supposed to be here.
        all_equal = np.all(present_classes == present_classes[0])  # Are all classes the same one?
        if all_equal:                               # If only one class present, then add it to the counter
            counter[present_classes[0]]["total"] += len(present_classes)
            counter[present_classes[0]]["solo images"] += 1
        else:               # If mixed case, then it must be balanced itself
            for cls in set(present_classes):
                counter[cls]["total"] += np.sum(present_classes == cls)
                counter[cls]["mixed"] += np.sum(present_classes == cls)
                counter[cls]["mixed images"] += 1
    min_case = np.min([counter[i]["total"] for i in range(label_patches.shape[-1]) if counter[i]["total"] != 0])
    for cls in range(label_patches.shape[-1]):
        assert counter[cls]["total"] == min_case or counter[cls]["total"] == counter[cls]["mixed"] or \
               counter[cls]["total"] == 0


def balance_test_segmentation(dataset_handler):
    list_ds = dataset_handler.get_dataset(method="separate",
                                          percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          balance_dataset=(True, True), stride=25,
                                          shuffle=True, classification=False, cast_to_np=True)
    verify_labels_balanced(list_ds[0][1])
    verify_labels_balanced(list_ds[1][1])
    list_ds = dataset_handler.get_dataset(method="random", balance_dataset=(True, True), stride=25,
                                          percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          shuffle=True, classification=False, cast_to_np=True)
    verify_labels_balanced(list_ds[0][1])
    verify_labels_balanced(list_ds[1][1])
    list_ds = dataset_handler.get_dataset(method="single_separated_image", balance_dataset=(True, True), stride=25,
                                          percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          shuffle=True, classification=False, cast_to_np=True)
    count = np.bincount(np.where(list_ds[0][1] == 1)[-1])
    assert np.all(np.logical_or(count == count[np.nonzero(count)][0], count == 0))
    count = np.bincount(np.where(list_ds[1][1] == 1)[-1])
    assert np.all(np.logical_or(count == count[np.nonzero(count)][0], count == 0))


def mode_change(dataset_handler):
    dataset_handler.mode = 't'
    try:
        dataset_handler.get_scattering_vector()
        raise Exception("Exception not catched")
    except NotImplementedError:
        pass  # This exception should be caught.
    assert dataset_handler.image.shape[-1] == 6
    dataset_handler.mode = 's'
    assert dataset_handler.image.shape[-1] == 3


def scattering_vector(dataset_handler):
    dataset_handler.mode = 'k'
    pauli_saved = dataset_handler.image
    s_image = dataset_handler.get_scattering_vector()
    dataset_handler.mode = 's'
    assert np.any(pauli_saved != dataset_handler.image)
    assert np.allclose(s_image, dataset_handler.image)


def test_tf_tensor_mode(dataset_handler):
    np_ds = dataset_handler.get_dataset(method="single_separated_image", percentage=1.)
    tf_ds = dataset_handler.get_dataset(method="single_separated_image", percentage=1.,  cast_to_np=True)
    compare_numpy_w_tf(np_ds, tf_ds)
    np_ds = dataset_handler.get_dataset(method="single_separated_image", percentage=(.3, .4, .3))
    tf_ds = dataset_handler.get_dataset(method="single_separated_image", percentage=(.3, .4, .3), cast_to_np=True)
    compare_numpy_w_tf(np_ds, tf_ds)
    if dataset_handler.name != "BRET":
        tf_ds = dataset_handler.get_dataset(method="random", percentage=1., shuffle=False, cast_to_np=True)
        np_ds = dataset_handler.get_dataset(method="random", percentage=1., shuffle=False)
        compare_numpy_w_tf(np_ds, tf_ds)
        tf_ds = dataset_handler.get_dataset(method="random", percentage=(0.5, .4), shuffle=False, cast_to_np=True)
        np_ds = dataset_handler.get_dataset(method="random", percentage=(0.5, .4), shuffle=False)
        compare_numpy_w_tf(np_ds, tf_ds)
        tf_ds = dataset_handler.get_dataset(method="separate", percentage=(0.5, .4), shuffle=False, cast_to_np=True)
        np_ds = dataset_handler.get_dataset(method="separate", percentage=(0.5, .4), shuffle=False)
        compare_numpy_w_tf(np_ds, tf_ds)
        tf_ds = dataset_handler.get_dataset(method="separate", percentage=1, shuffle=False, cast_to_np=True)
        np_ds = dataset_handler.get_dataset(method="separate", percentage=1, shuffle=False)
        compare_numpy_w_tf(np_ds, tf_ds)
        tf_ds = dataset_handler.get_dataset(method="separate", percentage=(0.5, .4, .1), shuffle=False, cast_to_np=True)
        np_ds = dataset_handler.get_dataset(method="separate", percentage=(0.5, .4, .1), shuffle=False)
        compare_numpy_w_tf(np_ds, tf_ds)


def compare_numpy_w_tf(np_ds, tf_ds):
    assert len(np_ds) == len(tf_ds)
    batch = 0
    for i in range(len(np_ds)):
        for j, elem in enumerate(iter(tf_ds[i])):
            last_batch = batch
            batch = elem[0].numpy().shape[0]
            assert np.all(elem[0].numpy() == np_ds[i][0][j*last_batch:j*last_batch+batch])
            assert np.all(elem[1].numpy() == np_ds[i][1][j*last_batch:j*last_batch+batch])


if __name__ == "__main__":
    # garon_balance_test(percentage=(0.8, 0.2))
    logging.basicConfig(level=logging.INFO)
    start_time = time.monotonic()
    test_bretigny()
    test_sf()
    test_flev()
    test_ober()
    logging.info(f"Time of tests {timedelta(seconds=time.monotonic() - start_time)}")