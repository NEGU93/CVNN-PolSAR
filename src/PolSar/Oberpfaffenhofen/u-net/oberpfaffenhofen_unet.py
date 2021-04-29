import numpy as np
from tensorflow.keras.layers import concatenate, Add, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, MaxPooling2D, UpSampling2D
from cvnn.layers import complex_input, ComplexConv2D, ComplexDropout, ComplexMaxPooling2D, ComplexUpSampling2D, \
    ComplexMaxPooling2DWithArgmax, ComplexUnPooling2D, ComplexInput
from cvnn.activations import softmax_real_with_avg, cart_relu
from cvnn.initializers import ComplexHeNormal

IMG_HEIGHT = 128
IMG_WIDTH = 128

cao_params_model = {
    'padding': 'same',
    'kernel_shape': (3, 3),                     # Table 1
    'block6_kernel_shape': (1, 1),              # Table 1 and paragraph after equation 12.
    'max_pool_kernel': (2, 2),                  # Table 1 and section 2.3.2 at 2nd paragraph
    'stride': 2,                                # Section 2.3.2 at 2nd paragraph
    'activation': cart_relu,                    # Equation 11 & 12
    'kernels': [12, 24, 48, 96, 192],           # Table 1
    'num_classes': 3,
    'dropout': 0.5,                             # TODO: Not found yet
    'output_function': softmax_real_with_avg,   # Section 2.3.2 at the end and section 2.4 (TODO: Not the same)
    'init': ComplexHeNormal,                    # Section 2.2
    'loss': 'category_crossentropy',            # Section 2.4
    'optimizer': Adam(learning_rate=0.0001, beta_1=0.9)
}


def get_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    dtype = np.complex64
    in1 = complex_input(shape=input_shape, dtype=dtype)
    # in1 = ComplexInput(input_shape=input_shape, dtype=dtype)

    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(in1)
    conv1 = ComplexDropout(0.5)(conv1)
    conv1 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv1)
    pool1 = ComplexMaxPooling2D((2, 2))(conv1)

    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool1)
    conv2 = ComplexDropout(0.5)(conv2)
    conv2 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv2)
    pool2 = ComplexMaxPooling2D((2, 2))(conv2)

    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool2)
    conv3 = ComplexDropout(0.5)(conv3)
    conv3 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv3)
    pool3 = ComplexMaxPooling2D((2, 2))(conv3)

    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(pool3)
    conv4 = ComplexDropout(0.5)(conv4)
    conv4 = ComplexConv2D(128, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv4)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv4)
    up1 = concatenate([tmp_up, conv3], axis=-1)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up1)
    conv5 = ComplexDropout(0.5)(conv5)
    conv5 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv5)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv5)
    up2 = concatenate([tmp_up, conv2], axis=-1)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv6 = ComplexDropout(0.5)(conv6)
    conv6 = ComplexConv2D(64, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv6)

    tmp_up = ComplexUpSampling2D((2, 2), dtype=dtype)(conv6)
    up2 = concatenate([tmp_up, conv1], axis=-1)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(up2)
    conv7 = ComplexDropout(0.5)(conv7)
    conv7 = ComplexConv2D(32, (3, 3), activation='cart_relu', padding='same', dtype=dtype)(conv7)
    segmentation = ComplexConv2D(3, (1, 1), activation='softmax_real_with_avg', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])
    losses = {'seg': 'categorical_crossentropy'}
    metrics = {'seg': ['acc']}
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    # https://github.com/tensorflow/tensorflow/issues/38988
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


def get_cao_cvfcn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    dtype = np.complex64

    def get_downsampling_block(input_to_block, num: int):
        conv = ComplexConv2D(cao_params_model['kernels'][num], cao_params_model['kernel_shape'],
                             activation='linear', padding=cao_params_model['padding'],
                             kernel_initializer=cao_params_model['init'], dtype=dtype)(input_to_block)
        # TODO: Add BN
        conv = Activation(cao_params_model['activation'])(conv)
        conv = ComplexDropout(cao_params_model['dropout'])(conv)
        pool, pool_argmax = ComplexMaxPooling2DWithArgmax(cao_params_model['max_pool_kernel'],
                                                          strides=cao_params_model['stride'])(conv)
        return pool, pool_argmax

    def get_upsampling_block(input_to_block, pool_argmax, desired_shape, kernels):
        # TODO: Shall I use dropout here too?
        unpool = ComplexUnPooling2D(desired_shape)([input_to_block, pool_argmax])
        conv = ComplexConv2D(kernels, cao_params_model['kernel_shape'],
                             activation=cao_params_model['activation'], padding=cao_params_model['padding'],
                             kernel_initializer=cao_params_model['init'], dtype=dtype)(unpool)
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
    conv7 = get_upsampling_block(conv6, pool5_argmax, pool4.shape[1:], cao_params_model['kernels'][3])     # Block7
    # Block 8
    add8 = Add()([conv7, pool4])
    conv8 = get_upsampling_block(add8, pool4_argmax, pool3.shape[1:], cao_params_model['kernels'][2])
    # Block 9
    add9 = Add()([conv8, pool3])
    conv9 = get_upsampling_block(add9, pool3_argmax, pool2.shape[1:], cao_params_model['kernels'][1])
    # Block 10
    add10 = Add()([conv9, pool2])
    conv10 = get_upsampling_block(add10, pool2_argmax, pool1.shape[1:], cao_params_model['kernels'][0])
    # Block 11
    add11 = Add()([conv10, pool1])
    conv11 = get_upsampling_block(add11, pool1_argmax, in1.shape[1:], cao_params_model['num_classes'])

    # Output
    out = Activation(cao_params_model['output_function'])(conv11)

    model = Model(inputs=[in1], outputs=[out])

    losses = {'seg': cao_params_model['loss']}
    metrics = {'seg': ['acc']}
    model.compile(optimizer=cao_params_model['optimizer'], loss=losses, metrics=metrics)

    # https://github.com/tensorflow/tensorflow/issues/38988
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    return model


if __name__ == '__main__':
    model_c = get_model()
    plot_model(model_c, show_shapes=True)
