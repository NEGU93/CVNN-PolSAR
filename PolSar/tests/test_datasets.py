import sys
import os
from os import makedirs
from pathlib import Path
import numpy as np
from dataset_readers.sf_data_reader import SanFranciscoDataset
from dataset_readers.flevoland_data_reader import FlevolandDataset
from dataset_readers.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from dataset_readers.bretigny_dataset import BretignyDataset
from dataset_readers.garon_dataset import GaronDataset
from pdb import set_trace
from cvnn.utils import create_folder

sys.path.insert(1, "/".join(os.path.abspath(__file__).split('/')[:-2]))
from principal_simulation import _get_dataset_handler, MODEL_META, DATASET_META


def test_coh_matrix_generator(kernel_shape=1):
    dataset_handler = SanFranciscoDataset(mode='t', dataset_name="SF-AIRSAR")
    coh = dataset_handler.get_image()
    dataset_handler.mode = 's'
    raw_s = dataset_handler.get_image()  # s_11, s_12, s_22
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
    for dataset_method in ["random", "separate", "single_separated_image"]:
        dataset_handler.get_dataset(method=dataset_method, percentage=(0.8, 0.2), size=128, stride=25, pad=0,
                                    shuffle=True, savefig=None, classification=False)
        rgb_image, ground_truth = verify_images(dataset_handler, ground_truth, rgb_image, show_gt, show_img)
    for dataset_method in ["separate", "random"]:
        dataset_handler.get_dataset(method=dataset_method, percentage=(0.8, 0.2), size=128, stride=25, pad=0,
                                    shuffle=True, savefig=None, classification=True)
        rgb_image, ground_truth = verify_images(dataset_handler, ground_truth, rgb_image, show_gt, show_img)


def test_sf(show_gt=False, show_img=False):
    dataset_handler = SanFranciscoDataset(mode='s', dataset_name="SF-AIRSAR")
    full_verify_dataset(dataset_handler)


def test_flev(show_gt=False, show_img=False):
    dataset_handler = FlevolandDataset(mode='s')
    full_verify_dataset(dataset_handler)


def test_ober(show_gt=False, show_img=False):
    dataset_handler = OberpfaffenhofenDataset(mode='s')
    full_verify_dataset(dataset_handler)


def test_bretigny(show_gt=False, show_img=False):
    dataset_handler = BretignyDataset(mode='s')
    full_verify_dataset(dataset_handler)


def full_verify_dataset(dataset_handler):
    balanced_classification_test(dataset_handler, percentage=(0.08, 0.02, 0.9), possible_to_balance=True)
    balance_test_segmentation(dataset_handler)
    handler_to_test(dataset_handler)
    mode_change(dataset_handler)
    scattering_vector(dataset_handler)


def balanced_classification_test(dataset_handler, percentage, possible_to_balance):
    # This method fails if I have the warning that the min samples was not met.
    list_ds = dataset_handler.get_dataset(method="random", percentage=percentage, size=6, stride=1, pad=0,
                                          shuffle=True, savefig=None, classification=True, balance_dataset=True)
    train_sparse = np.argmax(list_ds[0][1], axis=-1)
    train_count = np.bincount(train_sparse)
    assert np.all(train_count == train_count[0])
    val_sparse = np.argmax(list_ds[1][1], axis=-1)
    val_count = np.bincount(val_sparse)
    assert np.all(val_count == val_count[0])
    total = sum([list_ds[i][1].shape[0] for i in range(len(percentage))])
    for i, p in enumerate(percentage):
        if i != 0 or i != 1:
            assert np.isclose(p, list_ds[i][1].shape[0] / total, rtol=0.1)
    try:
        list_ds = dataset_handler.get_dataset(method="separate",
                                              percentage=DATASET_META[dataset_handler.name]["percentage"],
                                              size=6, stride=1, pad=0,
                                              shuffle=True, savefig=None, classification=True, balance_dataset=True)
    except Exception as e:
        if not possible_to_balance:
            return None
        else:
            raise e
    train_sparse = np.argmax(list_ds[0][1], axis=-1)
    train_count = np.bincount(train_sparse)
    assert np.all(train_count == train_count[0])


def garon_balance_test(percentage):
    dataset_handler = GaronDataset(mode='s', image_number=1)
    full_verify_dataset(dataset_handler)


def verify_labels_balanced(label_patches):
    """
    Verifies:
        - Same amount of 'one label only' images
        - 'Mixed-class' images are balanced
        - Total pixels per class balanced too
    Raises assertion error if image is not balanced
    :param label_patches:
    :return:
    """
    counter = np.zeros(len(label_patches))
    for i, la in enumerate(label_patches):
        present_classes = np.where(la == 1)[-1]     # Find all classes (again, there will be at least one).
        assert len(present_classes)                 # No empty image are supposed to be here.
        all_equal = np.all(present_classes == present_classes[0])  # Are all classes the same one?
        if all_equal:                               # If only one class present, then add it to the counter
            counter[i] = present_classes[0] + 1     # +1 because counter[i] = 0 is reserved for mix classes cases
        else:               # If mixed case, then it must be balanced itself
            occurrences = [np.sum(present_classes == cls) for cls in set(present_classes)]
            assert np.all(occurrences == occurrences[0])
    full_img_occurrences = np.bincount(counter.astype(int))
    assert np.all(full_img_occurrences[1:] == full_img_occurrences[1])      # Same amount of labeled images
    count = np.bincount(np.where(label_patches == 1)[-1])                   # Count of total pixels
    assert np.all(count == count[0])


def balance_test_segmentation(dataset_handler):
    # dataset_handler.balance_dataset = True
    list_ds = dataset_handler.get_dataset(method="separate",
                                          percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          balance_dataset=True, stride=25,
                                          shuffle=True, classification=False)
    verify_labels_balanced(list_ds[0][1])
    verify_labels_balanced(list_ds[1][1])
    list_ds = dataset_handler.get_dataset(method="random", percentage=DATASET_META[dataset_handler.name]["percentage"],
                                          balance_dataset=True, stride=25,
                                          shuffle=True, classification=False)
    verify_labels_balanced(list_ds[0][1])
    verify_labels_balanced(list_ds[1][1])
    # Single image is not balanced! (TODO: sth can be done with it)


def mode_change(dataset_handler):
    dataset_handler.mode = 't'
    try:
        dataset_handler.get_scattering_vector()
        raise Exception("Exception not catched")
    except NotImplementedError:
        pass    # This exception should be caught.
    assert dataset_handler.image.shape[-1] == 6
    dataset_handler.mode = 's'
    assert dataset_handler.image.shape[-1] == 3


def scattering_vector(dataset_handler):
    dataset_handler.mode = 'k'
    k_image = dataset_handler.get_scattering_vector()
    dataset_handler.mode = 's'
    assert np.allclose(k_image, dataset_handler.image)


if __name__ == "__main__":
    # garon_balance_test(percentage=(0.8, 0.2))
    test_bretigny()
    test_sf(show_gt=False, show_img=False)
    test_flev(False, False)
    test_ober()
    test_coh_matrix_generator()
