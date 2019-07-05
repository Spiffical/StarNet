from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.initializers import glorot_uniform, glorot_normal
from keras.layers.normalization import BatchNormalization

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


