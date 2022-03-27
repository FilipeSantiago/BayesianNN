import tensorflow as tf


class SimpleDense(tf.keras.layers.Layer):

    # add an activation parameter
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units

        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # pass the computation to the activation layer

        return self.activation(tf.matmul(inputs, self.w) + self.b)
