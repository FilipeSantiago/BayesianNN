import numpy as np
np.random.seed(7)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from layer.bayesian_dropout_layer import BayesianDropoutLayer
from layer.simple_dense import SimpleDense

n_samples = 20
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(2.5 * X), 0.1)

X_pred = np.atleast_2d(np.linspace(-3., 3., num=n_samples)).T

X = np.hstack((X, X ** 2, X ** 3))
X_pred = np.hstack((X_pred, X_pred ** 2, X_pred ** 3))

layer_1 = BayesianDropoutLayer(units=100, dropout=0.2, activation=tf.nn.relu)
layer_2 = BayesianDropoutLayer(units=100, dropout=0.2, activation=tf.nn.relu)
layer_3 = BayesianDropoutLayer(units=100, dropout=0.2, activation=tf.nn.relu)

model = Sequential()
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.add(Dense(1, activation=tf.identity))


def compute_loss(y, y_hat):
    regularization = layer_1.regularization + layer_2.regularization + layer_3.regularization
    loss = tf.reduce_sum(tf.square(tf.cast(y, tf.float32) - tf.cast(y_hat, tf.float32))) + regularization
    return loss


model.compile(loss=compute_loss, optimizer='adam', metrics=['mae'])

model.fit(X, y, epochs=1000, batch_size=32, verbose=2)

y_preds = []

for i in range(1000):
    res = model.predict(X_pred, batch_size=32)
    y_preds.append(res)

res_rscl = res
Y_rscl = y

if True:
    plt.Figure(figsize=(10, 8))
    plt.plot(X[:, 0], y, "r.")

    for pred in y_preds:
        plt.plot(X_pred[:, 0], pred, "b-", alpha=1./200)
    plt.grid()
    plt.show()
