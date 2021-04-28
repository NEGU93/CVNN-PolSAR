import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/barrachina/Documents/onera/src/PolSar/Oberpfaffenhofen')
from oberpfaffenhofen_dataset import get_dataset_for_segmentation
from oberpfaffenhofen_unet import get_model

if __name__ == "__main__":
    dataset = get_dataset_for_segmentation()

