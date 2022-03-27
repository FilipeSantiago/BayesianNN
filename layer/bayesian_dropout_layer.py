import tensorflow as tf


class BayesianDropoutLayer(tf.keras.layers.Layer):

    def __init__(self, units, dropout=0.1, l=1e-2, activation=tf.identity):
        super(BayesianDropoutLayer, self).__init__()
        self.units = units
        self.dropout = dropout
        self.l = l
        self.prob = 1. - dropout
        self.bernoulli = tf.compat.v1.distributions.Bernoulli(probs=self.prob, dtype=tf.float32)
        self.activation = activation

        self.n_in = None
        self.m = None
        self.M = None
        self.W = None

    def build(self, input_shape):
        self.n_in = input_shape[1]

        b_init = tf.zeros_initializer()
        self.m = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)

        w_init = tf.random_normal_initializer()
        self.M = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True)

        super().build(input_shape)

    def call(self, X):
        self.W = tf.matmul(tf.linalg.diag(self.bernoulli.sample((self.n_in,))), self.M)
        output = self.activation(tf.matmul(X, self.W) + self.m)
        return output

    @property
    def regularization(self):
        return self.l * (self.prob * tf.reduce_sum(tf.square(self.M)) + tf.reduce_sum(tf.square(self.m)))
