import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda

def gaussian_loss(sigma):
    """
    Custom NLL loss function which satisfies the 'proper scoring rule' as
    outlined in the paper Simple and Scalable Predictive Uncertainty Estimation
    using Deep Ensembles (https://arxiv.org/abs/1612.01474)

    # Arguments
        sigma: float, predicted variance.
        """

    def loss(y_true, y_pred):
        #return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 4
        div_result = Lambda(lambda x: x[0] / x[1])([K.square(y_true - y_pred), sigma])
        return K.mean(0.5*tf.log(sigma) + 0.5*div_result) + 5
    return loss
