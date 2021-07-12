import numpy as np
from tensorflow.keras.layers import concatenate, Add, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, \
    ComplexMaxPooling2DWithArgmax, ComplexUnPooling2D, ComplexInput, ComplexBatchNormalization, ComplexDense
from cvnn.activations import softmax_real_with_avg, cart_relu
from cvnn.initializers import ComplexHeNormal
from custom_accuracy import CustomCategoricalAccuracy

IMG_HEIGHT = None   # 128
IMG_WIDTH = None    # 128

cao_params_model = {
    'padding': 'same',
    'kernel_shape': (3, 3),                     # Table 1
    'block6_kernel_shape': (1, 1),              # Table 1 and paragraph after equation 12.
    'max_pool_kernel': (2, 2),                  # Table 1 and section 2.3.2 at 2nd paragraph
    'stride': 2,                                # Section 2.3.2 at 2nd paragraph
    'activation': cart_relu,                    # Equation 11 & 12
    'kernels': [12, 24, 48, 96, 192],           # Table 1
    'num_classes': 3,
    'dropout': 0.5,                             # TODO: Not found yet where
    'output_function': softmax_real_with_avg,   # Section 2.3.2 at the end and section 2.4
    'init': ComplexHeNormal(),                  # Section 2.2
    'loss': categorical_crossentropy,           # Section 2.4
    'optimizer': Adam(learning_rate=0.0001, beta_1=0.9)
}
cao_mlp_params = {
    'input_window': 32,
    'real_shape': [128, 256],
    'complex_shape': [96, 180]
}


def get_cao_cvfcn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.complex64):

    def get_downsampling_block(input_to_block, num: int):
        conv = ComplexConv2D(cao_params_model['kernels'][num], cao_params_model['kernel_shape'],
                             activation='linear', padding=cao_params_model['padding'],
                             kernel_initializer=cao_params_model['init'], dtype=dtype)(input_to_block)
        conv = ComplexBatchNormalization(dtype=dtype)(conv)
        conv = Activation(cao_params_model['activation'])(conv)
        conv = ComplexDropout(cao_params_model['dropout'])(conv)
        pool, pool_argmax = ComplexMaxPooling2DWithArgmax(cao_params_model['max_pool_kernel'],
                                                          strides=cao_params_model['stride'])(conv)
        return pool, pool_argmax

    def get_upsampling_block(input_to_block, pool_argmax, kernels,
                             activation=cao_params_model['activation'], dropout=True):
        # TODO: Shall I use dropout here too?
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
        conv = ComplexConv2D(kernels, cao_params_model['kernel_shape'],
                             activation='linear', padding=cao_params_model['padding'],
                             kernel_initializer=cao_params_model['init'], dtype=dtype)(unpool)
        conv = ComplexBatchNormalization(dtype=dtype)(conv)
        conv = Activation(activation)(conv)
        if dropout:
            conv = ComplexDropout(cao_params_model['dropout'])(conv)
        return conv

    in1 = complex_input(shape=input_shape)

    # Downsampling
    pool1, pool1_argmax = get_downsampling_block(in1, 0)        # Block 1
    pool2, pool2_argmax = get_downsampling_block(pool1, 1)      # Block 2
    pool3, pool3_argmax = get_downsampling_block(pool2, 2)      # Block 3
    pool4, pool4_argmax = get_downsampling_block(pool3, 3)      # Block 4
    pool5, pool5_argmax = get_downsampling_block(pool4, 4)      # Block 5

    # Bottleneck
    # Block 6
    conv6 = ComplexConv2D(cao_params_model['kernels'][4], (1, 1),
                          activation=cao_params_model['activation'], padding=cao_params_model['padding'],
                          dtype=dtype)(pool5)

    # Upsampling
    # Block7
    conv7 = get_upsampling_block(conv6, pool5_argmax, cao_params_model['kernels'][3])
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, pool4_argmax, cao_params_model['kernels'][2])
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, pool3_argmax, cao_params_model['kernels'][1])
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, pool2_argmax, cao_params_model['kernels'][0])
    # Block 11
    add11 = Add()([conv10, pool1])
    out = get_upsampling_block(add11, pool1_argmax, dropout=False,
                               kernels=cao_params_model['num_classes'], activation=cao_params_model['output_function'])

    model = Model(inputs=[in1], outputs=[out])
    model.compile(optimizer=cao_params_model['optimizer'], loss=cao_params_model['loss'],
                  metrics=[CustomCategoricalAccuracy(name='accuracy')])

    # https://github.com/tensorflow/tensorflow/issues/38988
    # model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


def get_tf_real_cao_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input
    import tensorflow as tf

    def get_downsampling_block(input_to_block, num: int):
        conv = Conv2D(cao_params_model['kernels'][num], cao_params_model['kernel_shape'],
                      activation='linear', padding=cao_params_model['padding'],
                      kernel_initializer="he_normal")(input_to_block)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(cao_params_model['dropout'])(conv)
        pool, pool_argmax = tf.nn.max_pool_with_argmax(conv, cao_params_model['max_pool_kernel'],
                                                       strides=cao_params_model['stride'], padding='VALID')
        return pool, pool_argmax

    def get_upsampling_block(input_to_block, pool_argmax, kernels,
                             activation="relu", dropout=True):
        # TODO: Shall I use dropout here too?
        unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])
        conv = Conv2D(kernels, cao_params_model['kernel_shape'],
                      activation='linear', padding=cao_params_model['padding'], kernel_initializer="he_normal")(unpool)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        if dropout:
            conv = Dropout(cao_params_model['dropout'])(conv)
        return conv

    in1 = Input(shape=input_shape)

    # Downsampling
    pool1, pool1_argmax = get_downsampling_block(in1, 0)  # Block 1
    pool2, pool2_argmax = get_downsampling_block(pool1, 1)  # Block 2
    pool3, pool3_argmax = get_downsampling_block(pool2, 2)  # Block 3
    pool4, pool4_argmax = get_downsampling_block(pool3, 3)  # Block 4
    pool5, pool5_argmax = get_downsampling_block(pool4, 4)  # Block 5

    # Bottleneck
    # Block 6
    conv6 = Conv2D(cao_params_model['kernels'][4], (1, 1),
                   activation="relu", padding=cao_params_model['padding'])(pool5)

    # Upsampling
    # Block7
    conv7 = get_upsampling_block(conv6, pool5_argmax, cao_params_model['kernels'][3])
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, pool4_argmax, cao_params_model['kernels'][2])
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, pool3_argmax, cao_params_model['kernels'][1])
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, pool2_argmax, cao_params_model['kernels'][0])
    # Block 11
    add11 = Add()([conv10, pool1])
    out = get_upsampling_block(add11, pool1_argmax, dropout=False,
                               kernels=cao_params_model['num_classes'], activation="softmax")

    model = Model(inputs=[in1], outputs=[out])
    model.compile(optimizer=cao_params_model['optimizer'], loss=cao_params_model['loss'],
                  metrics=[CustomCategoricalAccuracy(name='accuracy')])

    # https://github.com/tensorflow/tensorflow/issues/38988
    # model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


def get_cao_mlp_models(output_size, input_size=None):
    if input_size is None:
        input_size = cao_mlp_params['input_window']*cao_mlp_params['input_window']*6
    # Complex model
    shape = [ComplexInput(input_shape=input_size, dtype=np.complex64)]
    for i in range(0, len(cao_mlp_params['complex_shape'])):
        shape.append(ComplexDense(units=cao_mlp_params['complex_shape'][i], activation=cao_params_model['activation']))
        shape.append(ComplexDropout(rate=cao_params_model['dropout']))
    shape.append(ComplexDense(units=output_size, activation=cao_params_model['output_function']))
    complex_network = Sequential(name='cv-mlp', layers=shape)
    complex_network.compile(optimizer=cao_params_model['optimizer'], loss=cao_params_model['loss'],
                            metrics=['accuracy'])
    # Real model
    shape = [ComplexInput(input_shape=input_size*2, dtype=np.float32)]
    for i in range(0, len(cao_mlp_params['real_shape'])):
        shape.append(ComplexDense(units=cao_mlp_params['real_shape'][i], activation=cao_params_model['activation'],
                                  dtype=np.float32))
        shape.append(ComplexDropout(rate=cao_params_model['dropout'], dtype=np.float32))
    shape.append(ComplexDense(units=output_size, activation=cao_params_model['output_function'], dtype=np.float32))
    real_network = Sequential(name='cv-mlp', layers=shape)
    real_network.compile(optimizer=cao_params_model['optimizer'], loss=cao_params_model['loss'], metrics=['accuracy'])

    return [complex_network, real_network]



if __name__ == '__main__':
    model_c = get_tf_real_cao_model()
    plot_model(model_c, to_file="tf_model.png", show_shapes=True)
    # model_c = get_cao_cvfcn_model()
    # plot_model(model_c, to_file="cvnn_model.png", show_shapes=True)
