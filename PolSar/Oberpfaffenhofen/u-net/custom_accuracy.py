from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow import cast, bool
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend
import tensorflow.math as math
from pdb import set_trace


# def custom_categorical_accuracy(y_true, y_pred):
#     set_trace()
#     mask = reduce_std(math_ops.cast(y_true, backend.floatx()), axis=-1) == 0.
#     coincidences = math_ops.equal(math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1))
#     return math_ops.cast(coincidences or mask, backend.floatx())


class CustomAccuracy(Accuracy):

    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(CustomAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=cast(y_true, bool))


class CustomCategoricalAccuracy(CategoricalAccuracy):

    def __init__(self, name='custom_categorical_accuracy', **kwargs):
        super(CustomCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # I must mask when the result is [0, 0, 0]
        sample_weight = math.logical_not(math.reduce_all(math.logical_not(cast(y_true, bool)), axis=-1))
        super(CustomCategoricalAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


if __name__ == '__main__':
    y_true = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    y_pred = [[0.1, 0.9, 0.8], [0.1, 0.9, 0.8], [0.05, 0.95, 0]]

    set_trace()

    m = CustomCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    print(m.result().numpy())
