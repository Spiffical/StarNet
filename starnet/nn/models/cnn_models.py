import os, sys
sys.path.insert(0, os.getenv('HOME'))
import csv
import numpy as np

from starnet.nn.models.base_cnn import BaseModel
from starnet.nn.utilities.custom_layers import GaussianLayer
from StarNet.nn.utilities.custom_losses import gaussian_loss

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Input, Dropout, Activation
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam


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
                          activation=self._last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model


class StarNet2017DeepEnsemble(StarNet2017):

    def __init__(self):
        super(StarNet2017DeepEnsemble, self).__init__()
        self._model_type = 'StarNet2017_DeepEnsemble'
        self.loss_func = gaussian_loss

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

    def load_pretrained_model(self, model_path):

        model, sigma = self.model()
        model.load_weights(model_path)
        self.keras_model = model

        # If there is already a training log, collect latest learning rate
        self.fullfilepath = os.path.join(self.currentdir, self.folder_name)
        training_log_path = os.path.join(self.fullfilepath, 'training.log')
        if os.path.exists(training_log_path):
            with open(training_log_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                learning_rates = []
                for row in csv_reader:
                    learning_rates.append(float(row['lr']))
                if not np.isnan(learning_rates).all():  # Don't try to find min if all NaNs
                    self.lr = np.nanmin(learning_rates)
                print('Last learning rate of lr={} collected from: {}'.format(self.lr, training_log_path))
                self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                      clipnorm=1.)
        else:
            # TODO
            print('TODO')

        self.keras_model.compile(loss=self.loss_func(sigma), optimizer=self.optimizer, metrics=self.metrics)
        #self.keras_model = load_model(model,
        #                              custom_objects={'GaussianLayer': GaussianLayer,
        #                                              'loss': gaussian_loss})
        return None

    def compile_model(self):

        if self.keras_model is None:
            self.keras_model, sigma = self.model()
            self.keras_model.compile(loss=self.loss_func(sigma), optimizer=self.optimizer, metrics=self.metrics)

        return None

    
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
