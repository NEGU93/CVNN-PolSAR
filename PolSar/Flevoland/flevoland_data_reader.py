import scipy.io
import sys
from os import path
from notify_run import Notify
import traceback
from cvnn.montecarlo import MonteCarlo
if path.exists('/home/barrachina/Documents/onera/PolSar'):
    sys.path.insert(1, '/home/barrachina/Documents/onera/PolSar')
    dataset_path = "/media/barrachina/data/datasets/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = '/media/barrachina/data/datasets/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat'
    NOTIFY = False
elif path.exists('/usr/users/gpu-prof/gpu_barrachina/onera/PolSar'):
    sys.path.insert(1, '/usr/users/gpu-prof/gpu_barrachina/onera/PolSar')
    dataset_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Flevoland/AIRSAR_Flevoland/T3"
    labels_path = "/usr/users/gpu-prof/gpu_barrachina/datasets/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_15cls.mat"
    NOTIFY = True
else:
    raise FileNotFoundError("path of the oberpfaffenhofen dataset not found")
from dataset_reader import get_dataset_with_labels_t3, get_dataset_for_cao_segmentation, \
    get_dataset_for_cao_classification
from models.cao_fcnn import get_cao_mlp_models


def get_dataset_for_segmentation_cao():
    # flev_14 = scipy.io.loadmat(
    # '/media/barrachina/data/datasets/PolSar/Flevoland/AIRSAR_Flevoland/Label_Flevoland_14cls.mat')
    flev_15 = scipy.io.loadmat(labels_path)
    # labels_to_ground_truth(flev_15['label'], showfig=True)
    # labels_to_ground_truth(flev_14['label'], savefig="Flevoland_2")

    t3, labels = get_dataset_with_labels_t3(dataset_path=dataset_path, labels=flev_15['label'])
    return get_dataset_for_cao_segmentation(t3, labels)


def get_dataset_for_mlp():
    flev_15 = scipy.io.loadmat(labels_path)
    t3, labels = get_dataset_with_labels_t3(dataset_path=dataset_path, labels=flev_15['label'])
    return get_dataset_for_cao_classification(t3, flev_15['label'])


def train_mlp_models_montecarlo():
    x_train, x_test, y_train, y_test = get_dataset_for_mlp()
    models = get_cao_mlp_models(output_size=15)
    montecarlo = MonteCarlo()
    for model in models:
        montecarlo.add_model(model)
    montecarlo.run(x=x_train, y=y_train, validation_data=(x_test, y_test), iterations=10, epochs=200, batch_size=30,
                   debug=True)


def train_mlp_model():
    # https://notify.run/c/PGqsOzNQ1cSGdWM7
    if NOTIFY:
        notify = Notify()
        notify.send('Simulating Flevoland MLP models')
    try:
        time = train_mlp_models_montecarlo()
        if NOTIFY:
            notify.send(f"Simulations done in {time}")
    except Exception as e:
        if NOTIFY:
            notify.send("Error occurred")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    train_mlp_model()







