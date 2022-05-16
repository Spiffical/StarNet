import keras.backend as K
from keras import layers as ll
from keras.layers import Input, Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv1D, ZeroPadding1D, AveragePooling1D
from keras.initializers import glorot_uniform, glorot_normal
from keras.layers.normalization import BatchNormalization
from keras.engine import Input, Model
import tensorflow as tf

"""
Implementation of ResNet is a modified version retrieved from two sources:
https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/
https://github.com/viig99/mkscancer/blob/master/medcan_evaluate_next.py
"""


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    Implementation of the identity block

    Arguments:
    input_tensor -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    x -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    filters1, filters2, filters3 = filters

    # Save the input value. This is needed later to add back to the main path.
    x_shortcut = input_tensor

    # First component of main path
    x = Conv1D(filters=filters1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_normal(seed=0))(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv1D(filters=filters2, kernel_size=kernel_size, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_normal(seed=0))(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv1D(filters=filters3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_normal(seed=0))(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # Final step: Add shortcut value to main path, and pass it through a ReLU activation
    x = ll.Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, s=2):
    """
    Implementation of the convolutional block

    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    x -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    filters1, filters2, filters3 = filters

    # Save the input value
    x_shortcut = input_tensor

    ##### MAIN PATH #####
    # First component of main path
    x = Conv1D(filters1, 1, strides=s, name=conv_name_base + '2a',
               kernel_initializer=glorot_normal(seed=0))(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv1D(filters=filters2, kernel_size=kernel_size, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_normal(seed=0))(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv1D(filters=filters3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_normal(seed=0))(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    ##### SHORTCUT PATH ####
    x_shortcut = Conv1D(filters=filters3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_normal(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(name=bn_name_base + '1')(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a ReLU activation
    x = ll.Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise ValueError('Not a valid mode')


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    survival = K.random_binomial((1,), p=p_survival)
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y,
                           survival * y)


def stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1):
    """
    Stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf

    Residual block consisting of:
    - Conv - BN - ReLU - Conv - BN
    - identity shortcut connection
    - merge Conv path with shortcut path

    Original paper (http://arxiv.org/pdf/1512.03385v1.pdf) then has ReLU,
    but we leave this out: see https://github.com/gcr/torch-residual-networks

    Additional variants explored in http://arxiv.org/pdf/1603.05027v1.pdf

    some code adapted from https://github.com/dblN/stochastic_depth_keras
    """

    prev_nb_channels = K.int_shape(x)[2]

    if subsample_factor > 1:
        subsample = subsample_factor
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling1D(pool_size=subsample)(x)
        if nb_filters > prev_nb_channels:
            shortcut = Lambda(zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
    else:
        subsample = 1
        # shortcut: identity
        shortcut = x

    y = Conv1D(nb_filters, 3, strides=subsample, kernel_initializer='he_normal', padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(nb_filters, 3, strides=1, kernel_initializer='he_normal', padding='same')(y)
    y = BatchNormalization()(y)

    p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(y)

    out = ll.Add()([y, shortcut])

    return out


