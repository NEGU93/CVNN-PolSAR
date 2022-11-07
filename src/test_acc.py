import tensorflow as tf
import numpy as np
from cvnn.metrics import ComplexAverageAccuracy, ComplexCategoricalAccuracy
from qt_app import get_paths
from principal_simulation import open_saved_model, _get_dataset_handler, DATASET_META, MODEL_META
from pdb import set_trace

path = "/media/barrachina/data/results/During-Marriage-simulations/01Monday/run-15h50m19"

values = get_paths(path)[path]['params']


def _load_dataset(shuffle):
    if values['dataset_method'] == "random":
        percentage = MODEL_META[values['model']]["percentage"]
    else:
        percentage = DATASET_META[values['dataset']]["percentage"]
    dataset_handler = _get_dataset_handler(dataset_name=values['dataset'],
                                           mode='t' if values['dataset_mode'] == 'coh' else 's',
                                           complex_mode=True if values['dtype'] == 'complex' else False,
                                           real_mode=values['dtype'], normalize=False,
                                           balance=(values['balance'] == "dataset"))
    weights = dataset_handler.weights
    ds_list = dataset_handler.get_dataset(method=values['dataset_method'],
                                          task=MODEL_META[values['model']]["task"], percentage=percentage,
                                          size=MODEL_META[values['model']]["size"],
                                          stride=MODEL_META[values['model']]["stride"],
                                          pad=MODEL_META[values['model']]["pad"],
                                          shuffle=shuffle, savefig=False, data_augment=False,
                                          orientation=DATASET_META[values['dataset']]['orientation'],
                                          batch_size=MODEL_META[values['model']]['batch_size'])
    train_ds = ds_list[0]
    if len(ds_list) > 1:
        val_ds = ds_list[1]
    else:
        val_ds = None
    if len(ds_list) > 2:
        test_ds = ds_list[2]
    else:
        test_ds = None
    return train_ds, val_ds, weights


def _metric(metric, y_pred, y_true):
    metric.update_state(y_true, y_pred)
    print(f"{metric.name}: {metric.result().numpy()}")


def run_simu(shuffle):
    train_ds, val_ds, weights = _load_dataset(shuffle)
    model = open_saved_model(root_path=path, model_name=values['model'],
                             channels=6 if values['dataset_mode'] == "coh" else 3,
                             weights=weights if values['balance'] == "loss" else None, real_mode=values['dtype'],
                             num_classes=DATASET_META[values['dataset']]['classes'],
                             complex_mode=values['dtype'] == 'complex',
                             tensorflow=values['library'] == 'tensorflow')


    # print("evaluate before training")
    # model.evaluate(train_ds)
    # model.evaluate(val_ds)
    model.fit(train_ds, validation_data=val_ds, epochs=1)
    # print("evaluate after training")
    # model.evaluate(train_ds)
    # model.evaluate(val_ds)

    # set_trace()
    images = tf.convert_to_tensor(np.concatenate([x for x, y in train_ds], axis=0))
    labels = np.concatenate([y for x, y in train_ds], axis=0)
    y_pred_train = model.call(images, training=True)
    y_pred = model.call(images, training=False)
    m = tf.metrics.CategoricalAccuracy()
    print("CategoricalAccuracy:")
    _metric(m, y_pred=y_pred, y_true=labels)
    _metric(m, y_pred=y_pred_train, y_true=labels)
    m = ComplexCategoricalAccuracy()
    print("ComplexCategoricalAccuracy:")
    _metric(m, y_pred=y_pred, y_true=labels)
    _metric(m, y_pred=y_pred_train, y_true=labels)
    m = ComplexAverageAccuracy()
    print("ComplexAverageAccuracy:")
    _metric(m, y_pred=y_pred, y_true=labels)
    _metric(m, y_pred=y_pred_train, y_true=labels)


if __name__ == '__main__':
    run_simu(shuffle=True)
