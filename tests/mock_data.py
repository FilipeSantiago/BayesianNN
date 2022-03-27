import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer.bayesian_dropout_layer import BayesianDropoutLayer
from layer.simple_dense import SimpleDense
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model

n_samples = 2000
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(2.5 * X), 0.1).ravel()

X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T

X = np.hstack((X, X ** 2, X ** 3))
X_pred = np.hstack((X_pred, X_pred ** 2, X_pred ** 3))

n_feats = X.shape[1]
n_hidden = 20

auto_input = layers.Input(shape=(3))

# layer_1 = BayesianDropoutLayer(units=100, dropout=0.5, l=1e-2, activation=tf.nn.relu)
# layer_2 = BayesianDropoutLayer(units=100, dropout=0.5, l=1e-2, activation=tf.nn.relu)
# layer_3 = BayesianDropoutLayer(units=100, dropout=0.5, l=1e-2, activation=tf.nn.relu)
# layer_4 = BayesianDropoutLayer(units=1, dropout=0.5, l=1e-2)

layer_1 = layers.Dense(units=100, activation=tf.nn.relu)
layer_2 = layers.Dense(units=100, activation=tf.nn.relu)
layer_3 = layers.Dense(units=100, activation=tf.nn.relu)
layer_4 = layers.Dense(units=100, activation=tf.nn.relu)

layer_5 = layers.Dense(units=1)


layer_1_model = layer_1(auto_input)
layer_2_model = layer_2(layer_1_model)
layer_3_model = layer_3(layer_2_model)
layer_4_model = layer_4(layer_3_model)
layer_5_model = layer_5(layer_4_model)

model = Model(auto_input, layer_5_model)
model.summary()
optimizer = tf.keras.optimizers.Adam()


def compute_loss(y, y_hat):
    regularization = 0  # layer_1.regularization + layer_2.regularization + layer_3.regularization
    loss = tf.reduce_sum(tf.square(tf.cast(y, tf.float32) - tf.cast(y_hat, tf.float32))) + regularization
    return loss


@tf.function
def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        return y_hat, loss


for iter in range(1000):

    x_batch, y_batch = X, y
    y_hat, loss = train_step(x_batch, y_batch)

    if iter % 100 == 0:
        print("y", y, "y_hat", y_hat)

y_ = model.predict(X)

print(len(y_))

if True:
    plt.Figure(figsize=(10, 8))
    plt.plot(X[:, 0], y, "r.")
    plt.plot(X[:, 0], y_, "b.")
    plt.grid()
    plt.show()
