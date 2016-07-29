""" Sparse Auto Encoder Example.
This is a variant of https://github.com/aymericdamien/TensorFlow-Examples.
1. Sparsity constraint is added
2. The number of hidden layer is reduced to one.

Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
BETA = tf.constant(0.1)
RHO = tf.constant(0.05)

# Network Parameters
n_hidden_1 = 256 # Second layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_2

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_3

# Kullback-Leibler
def KL_divergence(rho, rho_hat):
  one_minus_rho = tf.sub(tf.constant(1.), rho)
  one_minus_rhohat = tf.sub(tf.constant(1.), rho_hat)
  divergence = tf.add(log_func(rho, rho_hat), log_func(one_minus_rho, one_minus_rhohat))
  return divergence 

def log_func(x, y):
  return tf.mul(x, tf.log(tf.div(x, y)))

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# sparse contraint
num_batch = tf.placeholder("float", 1)
rho_hat = tf.div(tf.reduce_sum(encoder_op, 0), num_batch)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost_sparse = tf.mul(BETA, tf.reduce_sum(KL_divergence(RHO, rho_hat)))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.add(cost, cost_sparse))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            #print(batch_xs[0][200])
            _, c, c_sparse = sess.run([optimizer, cost, cost_sparse], feed_dict={X: batch_xs, num_batch: np.array([batch_size])})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
