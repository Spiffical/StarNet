from keras import backend as K
from keras.layers import Layer
from keras.activations import softplus
from keras.initializers import glorot_normal


class GaussianLayer(Layer):
    """
    This custom Keras layer outputs both a predicted mu and sigma,
    allowing for uncertainty estimation.

    # Arguments
        output_dim: int, number of targets to predict.
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(GaussianLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = softplus(output_sig)
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]
