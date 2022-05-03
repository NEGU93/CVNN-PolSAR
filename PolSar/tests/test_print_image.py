import sys
from dataset_readers.bretigny_dataset import BretignyDataset
from dataset_readers.garon_dataset import GaronDataset
from dataset_readers.oberpfaffenhofen_dataset import OberpfaffenhofenDataset
from dataset_readers.sf_data_reader import SanFranciscoDataset
from dataset_readers.flevoland_data_reader import FlevolandDataset
sys.path.insert(1, '../')
from principal_simulation import DATASET_META


def test_generated_images(constructor, dataset_name):
    dataset_name = dataset_name.upper()
    dataset_handler = constructor(mode='s')
    dataset_handler.print_image_png(savefile=f"./generated_images/{dataset_name}", img_name=f"rgb_{dataset_name}_s.png")
    dataset_handler.print_ground_truth(path=f"./generated_images/{dataset_name}/labels_{dataset_name}_s.png",
                                       transparent_image=0.5)
    dataset_handler.get_dataset(method="single_separated_image", savefig=f"./generated_images/{dataset_name}/",
                                percentage=DATASET_META[dataset_name]["percentage"])
    dataset_handler = constructor(mode='t')
    dataset_handler.print_image_png(savefile=f"./generated_images/{dataset_name}", img_name=f"rgb_{dataset_name}_t.png")
    dataset_handler.print_ground_truth(path=f"./generated_images/{dataset_name}/labels_{dataset_name}_t.png",
                                       transparent_image=0.5)
    dataset_handler.get_dataset(method="single_separated_image", savefig=f"./generated_images/{dataset_name}/",
                                percentage=DATASET_META[dataset_name]["percentage"])


if __name__ == "__main__":
    test_generated_images(GaronDataset, dataset_name="GARON")
    test_generated_images(OberpfaffenhofenDataset, dataset_name="OBER")
    test_generated_images(SanFranciscoDataset, dataset_name="SF-AIRSAR")
    test_generated_images(FlevolandDataset, dataset_name="FLEVOLAND")
    test_generated_images(BretignyDataset, dataset_name="BRET")

