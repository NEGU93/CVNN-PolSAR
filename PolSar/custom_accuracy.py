from cvnn.metrics import ComplexAccuracy, ComplexCategoricalAccuracy, ComplexRecall, ComplexPrecision, ComplexCohenKappa
from tensorflow_addons.metrics import MeanMetricWrapper     # To have compat with tf v2.4, tf has MeanMetric since 2.6
from tensorflow import cast, bool
from tensorflow.python.keras import backend
import tensorflow as tf
from tensorflow import math
from pdb import set_trace


# def custom_categorical_accuracy(y_true, y_pred):
#     set_trace()
#     mask = reduce_std(math_ops.cast(y_true, backend.floatx()), axis=-1) == 0.
#     coincidences = math_ops.equal(math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1))
#     return math_ops.cast(coincidences or mask, backend.floatx())


class CustomAccuracy(ComplexAccuracy):

    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(CustomAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=cast(y_true, bool))


class CustomCategoricalAccuracy(ComplexCategoricalAccuracy):

    def __init__(self, name='custom_categorical_accuracy', **kwargs):
        super(CustomCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # I must mask when the result is [0, 0, 0]
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomCategoricalAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class CustomCohenKappa(ComplexCohenKappa):
    def __init__(self, num_classes, name='custom_cohen_kappa', **kwargs):
        super(CustomCohenKappa, self).__init__(num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # I must mask when the result is [0, 0, 0]
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomCohenKappa, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class CustomRecall(ComplexRecall):
    def __init__(self, name='custom_recall', **kwargs):
        super(CustomRecall, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomRecall, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class CustomPrecision(ComplexPrecision):
    def __init__(self, name='custom_precision', **kwargs):
        super(CustomPrecision, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomPrecision, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


def _accuracy(y_true, y_pred):
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    reduced_sum = tf.reduce_sum(tf.cast(tf.math.equal(y_true, y_pred), backend.floatx()), axis=-1)
    return tf.math.divide_no_nan(reduced_sum, tf.cast(tf.shape(y_pred)[-1], reduced_sum.dtype))


def custom_average_accuracy(y_true, y_pred):
    remove_zeros_mask = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
    y_true = tf.boolean_mask(y_true, remove_zeros_mask)
    y_pred = tf.boolean_mask(y_pred, remove_zeros_mask)
    num_cls = y_true.shape[-1]
    y_pred = math.argmax(y_pred, axis=-1)
    y_true = math.argmax(y_true, axis=-1)
    accuracies = []
    for i in range(0, num_cls):
        cls_mask = y_true == i
        # set_trace()
        accuracies.append(_accuracy(y_true=tf.boolean_mask(y_true, cls_mask),
                                    y_pred=tf.boolean_mask(y_pred, cls_mask)))
    accuracies = tf.convert_to_tensor(accuracies)
    return tf.math.reduce_sum(accuracies) / len(accuracies)


class CustomAverageAccuracy(MeanMetricWrapper):

    def __init__(self, name='custom_average_accuracy', dtype=None):
        super(CustomAverageAccuracy, self).__init__(custom_average_accuracy, name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomAverageAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


if __name__ == '__main__':
    y_true = [[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0], [0, 1, 0],
              [1, 0, 0]]
    y_pred = [[0.1, 0.9, 0.8],
              [0.1, 0.9, 0.8],
              [0.05, 0.95, 0], [0.95, 0.05, 0],
              [0, 1, 0]]

    m = CustomCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    print(m.result().numpy())
    print(custom_average_accuracy(y_true, y_pred).numpy())  # I want 0.5/3 = 1/6
