import os, sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
import csv
import numpy as np

from starnet.models.base import BaseModel, BaseDeepEnsemble, BaseDeepEnsembleTwoModelOutputs
from starnet.utils.nn_utils.custom_layers import GaussianLayer
from starnet.utils.nn_utils.resnet_architecture import identity_block, conv_block
from starnet.models.base import ModelMGPU

from keras.layers import Dense, Flatten, Input, Dropout, Activation
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.initializers import glorot_uniform, glorot_normal
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda


class StarNet2017(BaseModel):

    def __init__(self):
        super(StarNet2017, self).__init__()
        self._model_type = 'StarNet2017CNN'
        self.lr = 0.0007
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [256, 128]
        self.l2 = 0
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                              clipnorm=1.)
        self.last_layer_activation = 'linear'

    def model(self):
        input_tensor = Input(shape=self.get_input_shape(), name='input')
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(layer_3)
        layer_out = Dense(units=len(self.targetname), kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model


class StarNet2017DeepEnsemble(BaseDeepEnsemble):

    def __init__(self):
        super(StarNet2017DeepEnsemble, self).__init__()
        self._model_type = 'StarNet2017_DeepEnsemble'
        self.lr = 0.0007
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [256, 128]
        self.l2 = 0
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                              clipnorm=1.)
        self.last_layer_activation = 'linear'

    def model(self):
        input_tensor = Input(shape=self.get_input_shape(), name='input')
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(layer_3)
        mu, sigma = GaussianLayer(len(self.targetname), name='main_output')(layer_4)

        model = Model(inputs=input_tensor, outputs=mu)

        return model, sigma


class StarResNet(BaseModel):

    def __init__(self):
        super(StarResNet, self).__init__()
        self._model_type = 'StarNet_ResNet'
        self.last_layer_activation = 'linear'

    def model(self):
        """
        Implementation of the popular ResNet the following architecture:
        CONV1D -> BATCHNORM -> RELU -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        modified version retrieved from two sources:
            https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/
            https://github.com/viig99/mkscancer/blob/master/medcan_evaluate_next.py

        Returns:
        model -- a Model() instance in Keras
        """

        x_input = Input(self.get_input_shape())

        # Zero-Padding
        x = ZeroPadding1D(3)(x_input)

        # Stage 1
        x = Conv1D(4, 7, strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        #x = MaxPooling1D(3, strides=2)(x)

        # Stage 2
        x = conv_block(x, kernel_size=3, filters=[4, 4, 16], stage=2, block='a', s=1)
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='b')
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='c')

        # Stage 3
        x = conv_block(x, kernel_size=3, filters=[8, 8, 32], stage=3, block='a', s=2)
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='b')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='c')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='d')

        # Stage 4
        x = conv_block(x, kernel_size=3, filters=[16, 16, 64], stage=4, block='a', s=2)
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='c')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='d')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='e')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='f')

        # Stage 5
        x = conv_block(x, kernel_size=3, filters=[32, 32, 128], stage=5, block='a', s=2)
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='c')

        # AVGPOOL
        x = AveragePooling1D(2, name="avg_pool")(x)

        # Output layer
        x = Flatten()(x)
        x = Dense(len(self.targetname), activation=self.last_layer_activation, name='fc' + str(len(self.targetname)),
                  kernel_initializer=glorot_uniform(seed=0))(x)

        # Create model
        model = Model(inputs=x_input, outputs=x, name='ResNet')

        return model


class StarResNetSmall(BaseModel):

    def __init__(self):
        super(StarResNetSmall, self).__init__()
        self._model_type = 'StarNet_ResNet'
        self.last_layer_activation = 'linear'

    def model(self):
        """
        Implementation of the popular ResNet the following architecture:
        CONV1D -> BATCHNORM -> RELU -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        modified version retrieved from two sources:
            https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/
            https://github.com/viig99/mkscancer/blob/master/medcan_evaluate_next.py

        Returns:
        model -- a Model() instance in Keras
        """

        x_input = Input(self.get_input_shape())

        # Zero-Padding
        x = ZeroPadding1D(3)(x_input)

        # Stage 1
        x = Conv1D(4, 7, strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        # x = MaxPooling1D(3, strides=2)(x)

        # Stage 2
        x = conv_block(x, kernel_size=3, filters=[4, 4, 16], stage=2, block='a', s=1)
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='b')
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='c')

        # Stage 3
        x = conv_block(x, kernel_size=3, filters=[8, 8, 32], stage=3, block='a', s=2)
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='b')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='c')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='d')

        # Stage 4
        x = conv_block(x, kernel_size=3, filters=[16, 16, 64], stage=4, block='a', s=2)
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='c')

        # AVGPOOL
        x = AveragePooling1D(2, name="avg_pool")(x)

        # Output layer
        x = Flatten()(x)
        x = Dense(len(self.targetname), activation=self.last_layer_activation, name='fc' + str(len(self.targetname)),
                  kernel_initializer=glorot_uniform(seed=0))(x)

        # Create model
        model = Model(inputs=x_input, outputs=x, name='ResNet')

        return model


class StarResNetDeepEnsemble(BaseDeepEnsemble):

    def __init__(self):
        super(StarResNetDeepEnsemble, self).__init__()
        self._model_type = 'StarResNet_DeepEnsemble'

    def model(self):
        x_input = Input(self.get_input_shape())

        # Zero-Padding
        x = ZeroPadding1D(3)(x_input)

        # Stage 1
        x = Conv1D(4, 7, strides=1, name='conv1', kernel_initializer=glorot_normal(seed=0))(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        # x = MaxPooling1D(3, strides=2)(x)

        # Stage 2
        x = conv_block(x, kernel_size=3, filters=[4, 4, 16], stage=2, block='a', s=1)
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='b')
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='c')

        # Stage 3
        x = conv_block(x, kernel_size=3, filters=[8, 8, 32], stage=3, block='a', s=2)
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='b')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='c')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='d')

        # Stage 4
        x = conv_block(x, kernel_size=3, filters=[16, 16, 64], stage=4, block='a', s=2)
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='c')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='d')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='e')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='f')

        # Stage 5
        x = conv_block(x, kernel_size=3, filters=[32, 32, 128], stage=5, block='a', s=2)
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='c')

        # AVGPOOL
        x = AveragePooling1D(2, name="avg_pool")(x)

        # Output layer
        x = Flatten()(x)
        mu, sigma = GaussianLayer(len(self.targetname), name='main_output')(x)

        # Create model
        model = Model(inputs=x_input, outputs=mu, name='ResNetDeepEnsemble')

        return model, sigma


class StarResNetDeepEnsembleTwoOutputs(BaseDeepEnsembleTwoModelOutputs):

    def __init__(self):
        super(StarResNetDeepEnsembleTwoOutputs, self).__init__()
        self._model_type = 'StarResNet_DeepEnsemble'

    def model(self):
        x_input = Input(self.get_input_shape())

        # Zero-Padding
        x = ZeroPadding1D(3)(x_input)

        # Stage 1
        x = Conv1D(4, 7, strides=1, name='conv1', kernel_initializer=glorot_normal(seed=0))(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        # x = MaxPooling1D(3, strides=2)(x)

        # Stage 2
        x = conv_block(x, kernel_size=3, filters=[4, 4, 16], stage=2, block='a', s=1)
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='b')
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='c')

        # Stage 3
        x = conv_block(x, kernel_size=3, filters=[8, 8, 32], stage=3, block='a', s=2)
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='b')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='c')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='d')

        # Stage 4
        x = conv_block(x, kernel_size=3, filters=[16, 16, 64], stage=4, block='a', s=2)
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='c')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='d')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='e')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='f')

        # Stage 5
        x = conv_block(x, kernel_size=3, filters=[32, 32, 128], stage=5, block='a', s=2)
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')
        x = identity_block(x, 3, [32, 32, 128], stage=5, block='c')

        # AVGPOOL
        x = AveragePooling1D(2, name="avg_pool")(x)

        # Output layer
        x = Flatten()(x)
        mu, sigma = GaussianLayer(len(self.targetname), name='main_output')(x)

        # Additional 'input' for the labels
        label_layer = Input((len(self.targetname),))

        # Create model
        model = Model(inputs=[x_input, label_layer], outputs=[mu, sigma], name='ResNetDeepEnsemble')

        # Define the loss function (needs to be defined here because it uses an intermediate layer)
        # NOTE: do not include loss function when compiling model because of this
        div_result = Lambda(lambda y: y[0] / y[1])([K.square(label_layer - mu), sigma])
        loss = K.mean(0.5 * tf.log(sigma) + 0.5 * div_result) + 5

        # Add loss to model
        model.add_loss(loss)

        return model


class StarResNetSmallDeepEnsemble(BaseDeepEnsemble):

    def __init__(self):
        super(StarResNetSmallDeepEnsemble, self).__init__()
        self._model_type = 'StarResNetSmall_DeepEnsemble'

    def model(self):
        x_input = Input(self.get_input_shape())

        # Zero-Padding
        x = ZeroPadding1D(3)(x_input)

        # Stage 1
        x = Conv1D(4, 7, strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        # x = MaxPooling1D(3, strides=2)(x)

        # Stage 2
        x = conv_block(x, kernel_size=3, filters=[4, 4, 16], stage=2, block='a', s=1)
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='b')
        x = identity_block(x, 3, [4, 4, 16], stage=2, block='c')

        # Stage 3
        x = conv_block(x, kernel_size=3, filters=[8, 8, 32], stage=3, block='a', s=2)
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='b')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='c')
        x = identity_block(x, 3, [8, 8, 32], stage=3, block='d')

        # Stage 4
        x = conv_block(x, kernel_size=3, filters=[16, 16, 64], stage=4, block='a', s=2)
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=4, block='c')

        # AVGPOOL
        x = AveragePooling1D(2, name="avg_pool")(x)

        # Output layer
        x = Flatten()(x)
        mu, sigma = GaussianLayer(len(self.targetname), name='main_output')(x)

        # Create model
        model = Model(inputs=x_input, outputs=mu, name='ResNetDeepEnsemble')

        return model, sigma


class StarResNet_old(BaseModel):

    def __init__(self):
        super(StarResNet_old, self).__init__()
        self.last_layer_activation = 'linear'

    def model(self):
        """
        Implementation of the popular ResNet
        https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314

        Returns:
        model -- a Model() instance in Keras
        """
        from keras.layers.merge import add
        def block(n_output, upscale=False):
            # n_output: number of feature maps in the block
            # upscale: should we use the 1x1 conv2d mapping for shortcut or not

            # keras functional api: return the function of type
            # Tensor -> Tensor
            def f(x):

                # H_l(x):
                # first pre-activation
                h = BatchNormalization()(x)
                h = Activation('relu')(h)
                # first convolution
                h = Conv1D(kernel_size=3, filters=n_output, strides=1, padding='same')(h)

                # second pre-activation
                h = BatchNormalization()(h)
                h = Activation('relu')(h)
                # second convolution
                h = Conv1D(kernel_size=3, filters=n_output, strides=1, padding='same')(h)

                # f(x):
                if upscale:
                    # 1x1 conv2d
                    f = Conv1D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
                else:
                    # identity
                    f = x

                # F_l(x) = f(x) + H_l(x):
                return add([f, h])

            return f


        x_input = Input(self.get_input_shape())

        # first conv2d with post-activation to transform the input data to some reasonable form
        x = Conv1D(kernel_size=3, filters=4, strides=1, padding='same')(x_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # F_1
        x = block(4)(x)
        # F_2
        x = block(4)(x)

        # F_3
        # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
        # and we can't add together tensors of inconsistent sizes, so we use upscale=True
        x = block(8, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_4
        x = block(8)(x)                     # !!! <------- Uncomment for local evaluation
        # F_5
        x = block(8)(x)                     # !!! <------- Uncomment for local evaluation

        # F_6
        x = block(16, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_7
        x = block(16)(x)                     # !!! <------- Uncomment for local evaluation

        # last activation of the entire network's output
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # average pooling across the channels
        # 28x28x48 -> 1x48
        x = AveragePooling1D()(x)

        # dropout for more robust learning
        #x = Dropout(0.2)(x)

        # output layer
        x = Flatten()(x)
        x = Dense(len(self.targetname), activation=self.last_layer_activation, name='fc' + str(len(self.targetname)),
                  kernel_initializer=glorot_uniform(seed=0))(x)

        # Create model
        model = Model(inputs=x_input, outputs=x, name='ResNet')

        return model

    
class BCNN(StarNet2017):
    """
    Class for Bayesian convolutional neural network for stellar spectra analysis
    :History: 2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.0005, dropout_rate=0.3):
        super(BCNN, self).__init__()

        self.name = 'BCNN'
        self._model_type = 'BCNN'
        self._implementation_version = '1.0'
        self.initializer = 'he_normal'#RandomNormal(mean=0.0, stddev=0.05)
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = 'regression'
        
        self.disable_dropout = False

    def model(self):
        input_tensor = Input(shape=self.input_shape, name='input')

        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = Dropout(self.dropout_rate)(activation_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = Dropout(self.dropout_rate)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = Dropout(self.dropout_rate)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        output = Dense(units=len(self.targetname), activation=self._last_layer_activation, name='output')(activation_4)

        model = Model(inputs=input_tensor, outputs=output)
        return model


if __name__ == '__main__':
    print('main')
