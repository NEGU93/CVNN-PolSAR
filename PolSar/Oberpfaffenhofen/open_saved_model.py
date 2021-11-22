import sys
import numpy as np
sys.path.insert(1, 'W:\HardDiskDrive\Documentos\GitHub\onera\PolSar')
from models.cao_fcnn import get_cao_cvfcn_model, get_tf_real_cao_model
from oberpfaffenhofen_dataset import get_ober_dataset_with_labels_t6
from dataset_reader import labels_to_rgb

def get_saved_models(checkpoint_path, complex_mode=True, tensorflow=False):
    if not tensorflow:
        if complex_mode:
            model = get_cao_cvfcn_model(input_shape=(128, 128, 21))
        else:
            model = get_cao_cvfcn_model(input_shape=(128, 128, 42),
                                        dtype=np.float32)
    else:
        if complex_mode:
            raise ValueError("Tensorflow does not support complex model. "
                             "Do not use tensorflow and complex_mode both as True")
        model = get_tf_real_cao_model(input_shape=(128, 128, 42))
    
    model.load_weights(checkpoint_path)
    return model


if __name__ == "__main__":
    model = get_saved_models("W:\HardDiskDrive\Documentos\GitHub\onera\PolSar\Oberpfaffenhofen\log\\2021\\06June\\18Friday\\run-11h38m05\checkpoints\cp.ckpt")
    size = 128
    T, labels = get_ober_dataset_with_labels_t6()
    predicted = np.zeros(T.shape[:2] + (3,))
    slices = []
    slices_labels = []
    
    for i in range(0, int(T.shape[0]/size)):
        for j in range(0, int(T.shape[1]/size)):
            slice_x = slice(i*size, (i+1)*size)
            slice_y = slice(j*size, (j+1)*size)
            slices.append(T[slice_x, slice_y].astype(np.complex64))
            slices_labels.append(labels[slice_x, slice_y])
            sliced_matrix = slices[-1].reshape(1, size, size, 21)
            predicted_slice = model(sliced_matrix, training=False)[0].numpy()
            predicted[slice_x, slice_y] = predicted_slice     
    slices = np.array(slices)
    slices_labels = np.array(slices_labels)
    # print(f"Evaluate {model.evaluate(x=slices, y=slices_labels)}")            
    # set_trace()
    labels_to_rgb(predicted, showfig=True, savefig="W:\HardDiskDrive\Documentos\GitHub\onera\PolSar\Oberpfaffenhofen\log\\2021\\06June\\18Friday\\run-11h38m05\prediction")