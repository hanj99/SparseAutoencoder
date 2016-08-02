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
training_epochs = 10 
batch_size = 256 
display_step = 1
examples_to_show = 10
BETA = tf.constant(0.01)
RHO = tf.constant(0.05)

# [n_input, n_hidden_layer1, n_hidden_layer2, ... n_output]
n_units_per_layer = [784, 256, 256, 256, 10]
n_layers = len(n_units_per_layer) 

# tf Graph input (only pictures)
Xs = []
weights = {}
biases = {}
for layer_idx in range(n_layers)):
  Xs.append(tf.placeholder("float", [None, n_units_per_layer[layer_idx]]))

for layer_idx in range(n_layers-1):
  weights["encoder_h{0}".format(layer_idx+1)] = tf.Variable(tf.truncated_normal([n_units_per_layer[layer_idx], n_units_per_layer[layer_idx + 1]], stddev=0.1))
  weights["decoder_h{0}".format(layer_idx+1)] = tf.Variable(tf.truncated_normal([n_units_per_layer[layer_idx+1], n_units_per_layer[layer_idx]], stddev=0.1))
  biases["encoder_b{0}".format(layer_idx+1)] = tf.Variable(tf.constant(0.1, shape=[n_units_per_layer[layer_idx+1]]))
  biases["decoder_b{0}".format(layer_idx+1)] = tf.Variable(tf.constant(0.1, shape=[n_units_per_layer[layer_idx]]))

# Building the encoder
def encoder(x, layer_no):
    activation = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h%d' % layer_no]), biases['encoder_b%d' % layer_no]))
    return activation 

# Building the decoder
def decoder(y, layer_no):
    activation = tf.nn.sigmoid(tf.add(tf.matmul(y, weights['decoder_h%d' % layer_no]), biases['decoder_b%d' % layer_no]))
    return activation 

# Kullback-Leibler
def KL_divergence(rho, rho_hat):
  one_minus_rho = tf.sub(tf.constant(1.), rho)
  one_minus_rhohat = tf.sub(tf.constant(1.), rho_hat)
  divergence = tf.add(log_func(rho, rho_hat), log_func(one_minus_rho, one_minus_rhohat))
  return divergence 

def log_func(x, y):
  return tf.mul(x, tf.log(tf.div(x, y)))

# Construct model
encoders = []
decoders = []

for layer_idx in range(n_layers-2):
  encoders.append(encoder(Xs[layer_idx], layer_idx+1))
  decoders.append(decoder(encoders[layer_idx], layer_idx+1))

# for fine tunning
encoder_chains = {"encoder_l0": Xs[0]} 
for layer_idx in range(n_layers-1):
  encoder_chains["encoder_l{0}".format(layer_idx+1)] = encoder(encoder_chains["encoder_l{0}".format(layer_idx)], layer_idx+1)

decoder_chains = {"decoder_l{0}".format(n_layers): encoder_chains["encoder_l{0}".format(n_layers-1)]}
for layer_idx in range(n_layers-1, 0, -1):
  decoder_chains["decoder_l{0}".format(layer_idx)] = decoder(decoder_chains["decoder_l{0}".format(layer_idx+1)], layer_idx)


# sparse contraint
num_batch = tf.placeholder("float", 1)
rho_hats = []
for i in range(n_layers-2):
  rho_hats.append(tf.div(tf.reduce_sum(encoders[i], 0), num_batch))

# Prediction
y_preds = []
# Targets (Labels) are the input data.
y_trues = []
for i in range(n_layers-2):
  y_preds.append(decoders[i])
  y_trues.append(Xs[i])

# Classification layer
Y = tf.nn.softmax(encoder_chains["encoder_l{0}".format(n_layers-1)]) 
Y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_sum(- Y_ * tf.log(Y) - (1 - Y_) * tf.log(1 - Y), reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Define loss and optimizer, minimize the squared error
costs = []
costs_sparse = []
optimizers = []
for i in range(n_layers-2):
  costs.append(tf.reduce_mean(tf.pow(y_trues[i] - y_preds[i], 2)))
  costs_sparse.append(tf.mul(BETA, tf.reduce_sum(KL_divergence(RHO, rho_hats[i]))))
  optimizers.append(tf.train.RMSPropOptimizer(learning_rate).minimize(tf.add(costs[i], costs_sparse[i])))

#fine tunning
cost = tf.reduce_mean(tf.pow(Xs[0] - decoder_chains["decoder_l1"], 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)

    for layer_idx in range(n_layers-2):
      # Training cycle
      for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
          if layer_idx > 0:
            for j in range(layer_idx):
              batch_xs = sess.run(encoders[j], feed_dict={Xs[j]: batch_xs})

          # Run optimization op (backprop) and cost op (to get loss value)
          _, c, c_sparse = sess.run([optimizers[layer_idx], costs[layer_idx], costs_sparse[layer_idx]], feed_dict={Xs[layer_idx]: batch_xs, num_batch: np.array([batch_size])})

        #Display logs per epoch step
        if epoch % display_step == 0:
          print("Layer:", '%04d' % (layer_idx+1),
                "Epoch:", '%04d' % (epoch+1),
                "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # classification training
    for i in range(2000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      train_step.run({Xs[0]: batch_xs, Y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({Xs[0]: mnist.test.images, Y_: mnist.test.labels}))
