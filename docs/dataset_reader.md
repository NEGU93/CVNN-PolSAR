# Dataset Reader

## Usage

To use this mudule you must inherit implement two methods:

- `get_image`: Must open the image. It must be:
    - numpy array
    - Data type `np.complex`
    - Shape (Width, Height, channels), with `channels = 3` if `self.mode = 'k'` or `'s'`
        and `channels = 6` if `self.mode = 't'`.
        S format: (s_11, s_12, s_22) or equivalently (HH, HV, VV)
        T format:
    :return: `np.ndarray` The opened numpy image.

- get_sparse_labels: Must open the labels in sparse mode (last dimension is a number from 0 to `num_classes`).
    ATTENTION: Class 0 is considered as an unlabeled pixel.
    :return: Numpy array with the sparse labels

Attention: An internal variable ``self.azimuth` is advised to be defined in the constructor to tell the class the azimuth direction.

### Example

```
from dataset_reader import PolsarDatasetHandler

root_path = "path/to/dataset/folder"
labels_path = "path/to/labels/folder"

class MyDatasetReader(PolsarDatasetHandler):

    def __init__(self, *args, **kwargs):
        super(FlevolandDataset, self).__init__(root_path=os.path.dirname(labels_path),
                                               name="My Dataset", *args, **kwargs)
        self.azimuth = "horizontal"

    def get_image(self):
        if self.mode == "s":
            data = self.open_s_dataset(str(Path(root_path) / "S2"))
        elif self.mode == "t":
            data = self.open_t_dataset_t3(str(Path(root_path) / "T3"))
        elif self.mode == "k":
            mat = self.open_s_dataset(str(Path(root_path) / "S2"))    # s11, s12, s22
            data = self._get_k_vector(HH=mat[:, :, 0], VV=mat[:, :, 2], HV=mat[:, :, 1])
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

    def get_sparse_labels(self):
        return scipy.io.loadmat(labels_path)['label']
```

**Note**: `open_s_dataset` and `open_t_dataset_t3` methods can be used to open PolSARpro default output.

## Attributes

- image: 3D complex data 
- labels: One-hot-encoded labels
- sparse_labels: Sparse labels
- labels_occurrences: List of len equal to the number of classes of ratio frequency of each class.

## Methods

```
get_dataset(): Get the dataset and labels in the desired form
    :param method: One of
        - 'random': Sample patch images randomly using sliding window operation (swo).
        - 'separate': Splits the image according to `percentage` parameter. Then gets patches using swo.
        - 'single_separated_image': Splits the image according to `percentage` parameter. Returns full image.
    :param percentage: Tuple giving the dataset split percentage.
        If sum(percentage) != 1 it will add an extra value to force `sum(percentage) = 1`.
        If sum(percentage) > 1 or it has at least one negative value it will raise an exception.
        Example, for 60% train, 20% validation and 20% test set, use `percentage = (.6, .2, .2)` or `(.6, .2)`.
    :param size: Size of generated patches images. By default it will generate images of `128x128`.
    :param stride: Stride used for the swo. If `stride < size`, parches will have coincident pixels.
    :param shuffle: Shuffle image patches (ignored if `method == 'single_separated_image'`)
    :param pad: Pad image before swo or just add padding to output for `method == 'single_separated_image'`
    :param savefig: Used only if `method='single_separated_image'`.
        - It shaves `len(percentage)` images with the cropped generated images.
    :param azimuth: Cut the image 'horizontally' or 'vertically' when split (using percentage param for sizes).
        Ignored if `method == 'random'`
    :param data_augment: Only used if `use_tf_dataset = True`. It performs data aumentation using flip.
    :param remove_last:
    :param classification: If `True`, it will have only one value per image path.
        Example, for a train dataset of shape `(None, 128, 128, 3)`:
            classification = `True`: labels will be of shape `(None, classes)`
            classification = `False`: labels will be of shape `(None, 128, 128, classes)`
    :param complex_mode: (default = `True`). Whether to return the data in complex dtype or float.
    :param real_mode: If `complex_mode = False`, this param is used to specify the float format. One of:
        - real_imag: Stack real and imaginary part
        - amplitude_phase: stack amplitude and phase
        - amplitude_only: output only the amplitude
        - real_only: output only the real part
    :param batch_size: Used only if `use_tf_dataset = True`. Fixes the batch size of the tf.Dataset
    :param use_tf_dataset: If `True`, return dtype will be a `tf.Tensor` dataset instead of numpy array.
    :return: Returns a list of `[train, (validation), (test), (k-folds)]` according to percentage parameter.
        - Each `list[i]` is a tuple of `(data, labels)` where both data and labels are numpy arrays.
```
```
print_ground_truth: Saves or shows the labels rgb map.
    :param label: Labels to be printed as RGB map. If None it will use the dataset labels.
    :param path: Path where to save the image
    :param transparent_image: One of:
        - If `True` it will also print the rgb image to superpose with the labels.
        - float: alpha value for the plotted image (if `True` it will use default)
    :param mask: (Optional) One of
        - Boolean array with the same shape as label. `False` values will be printed as black.
        - If `True`: It will use self label to remove non labeled pixels from images
        - Ignored if label is None
    :param ax: (Optional) axis where to plot the new image, used for overlapping figures.
    :param showfig: Show figure
    :return: np array of the rgb ground truth image
```
```
print_image_png: Generates the RGB image
    :param savefile: Where to save the image or not.
        - Bool: If True it will save the image at self.root_path
        - str: path where to save the image
    :param showfig: Show image
    :param img_name: Name of the generated image
    :return: Rge rgb image as numpy
```
```
get_occurrences: Get the occurrences of each label
    :param labels: (Optional) if `None` it will return the occurrences of self labels.
    :param normalized: Normalized the output, for example, [20, 10] will be transformed to [2, 1]
        - This is used to obtain the weights of a penalized loss.
    :return: a list label-wise occurrences
```


