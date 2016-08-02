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
n_hidden_1 = 128 
n_hidden_2 = 128 
n_hidden_3 = 10 
n_input = 784 # MNIST data input (img shape: 28*28)
n_layers = 2 

# tf Graph input (only pictures)
Xs = []
Xs.append(tf.placeholder("float", [None, n_input]))
Xs.append(tf.placeholder("float", [None, n_hidden_1]))
Xs.append(tf.placeholder("float", [None, n_hidden_2]))
Xs.append(tf.placeholder("float", [None, n_hidden_3]))
Ys = []
Ys.append(tf.placeholder("float", [None, n_hidden_1]))
Ys.append(tf.placeholder("float", [None, n_hidden_2]))
Ys.append(tf.placeholder("float", [None, n_hidden_3]))

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.1)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
}
biases = {
    'encoder_b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'encoder_b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'encoder_b3': tf.Variable(tf.constant(0.1, shape=[n_hidden_3])),
    'decoder_b1': tf.Variable(tf.constant(0.1, shape=[n_input])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
}


# Building the encoder
def encoder(x, layer_no):
    #activation = tf.nn.softmax(tf.add(tf.matmul(x, weights['encoder_h%d' % layer_no]), biases['encoder_b%d' % layer_no]))
    activation = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h%d' % layer_no]), biases['encoder_b%d' % layer_no]))
    #activation = (tf.add(tf.matmul(x, weights['encoder_h%d' % layer_no]), biases['encoder_b%d' % layer_no]))
    return activation 

# Building the decoder
def decoder(y, layer_no):
    activation = tf.nn.sigmoid(tf.add(tf.matmul(y, weights['decoder_h%d' % layer_no]),
                                   biases['decoder_b%d' % layer_no]))
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
decoders_final = []
for i in range(n_layers):
  encoders.append(encoder(Xs[i], i+1))
  decoders.append(decoder(encoders[i], i+1))
  decoders_final.append(decoder(Ys[i], i+1))

# for fine tunning
encoder_l1 = encoder(Xs[0], 1)
encoder_l2 = encoder(encoder_l1, 2)
encoder_l3 = encoder(encoder_l2, 3)
decoder_l3 = decoder(encoder_l3, 3)
decoder_l2 = decoder(decoder_l3, 2)
decoder_l1 = decoder(decoder_l2, 1)

# sparse contraint
num_batch = tf.placeholder("float", 1)
rho_hats = []
for i in range(n_layers):
  rho_hats.append(tf.div(tf.reduce_sum(encoders[i], 0), num_batch))

# Prediction
y_preds = []
# Targets (Labels) are the input data.
y_trues = []
for i in range(n_layers):
  y_preds.append(decoders[i])
  y_trues.append(Xs[i])

# Classification layer
#Y = tf.nn.softmax(encoder_l3) #FIXME
Y = tf.nn.softmax(encoder_l3) #FIXME
Y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = -1 * tf.reduce_mean(tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
cross_entropy = tf.reduce_sum(- Y_ * tf.log(Y) - (1 - Y_) * tf.log(1 - Y), reduction_indices=[1])
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(encoder_l3, Y_) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Define loss and optimizer, minimize the squared error
costs = []
costs_sparse = []
optimizers = []
for i in range(n_layers):
  costs.append(tf.reduce_mean(tf.pow(y_trues[i] - y_preds[i], 2)))
  costs_sparse.append(tf.mul(BETA, tf.reduce_sum(KL_divergence(RHO, rho_hats[i]))))
  optimizers.append(tf.train.RMSPropOptimizer(learning_rate).minimize(tf.add(costs[i], costs_sparse[i])))

#fine tunning
cost = tf.reduce_mean(tf.pow(Xs[0] - decoder_l1, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)

    for layer_idx in range(n_layers):
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

    # fine tunning 
    #for epoch in range(training_epochs):
    #  for i in range(total_batch):
    #    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #    _, c = sess.run([optimizer, cost], feed_dict={Xs[0]: batch_xs, num_batch: np.array([batch_size])})

    #  if epoch % display_step == 0:
    #    print("Layer:", '%04d' % (layer_idx+1),
    #          "Epoch:", '%04d' % (epoch+1),
    #          "cost=", "{:.9f}".format(c))

    # Applying encode and decode over test set
    #batch_xs = mnist.test.images[:examples_to_show]
    #for layer_idx in range(n_layers):
    #  batch_xs = sess.run(encoders[layer_idx], feed_dict={Xs[layer_idx]: batch_xs})

    #batch_ys = batch_xs
    #for layer_idx in range(n_layers-1, -1, -1):
    #  batch_ys = sess.run(decoders_final[layer_idx], feed_dict={Ys[layer_idx]: batch_ys})

    ## Compare original images with their reconstructions
    #f, a = plt.subplots(2, 10, figsize=(10, 2))
    #for i in range(examples_to_show):
    #    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #    a[1][i].imshow(np.reshape(batch_ys[i], (28, 28)))
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()


    # classification training
    for i in range(2000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      train_step.run({Xs[0]: batch_xs, Y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({Xs[0]: mnist.test.images, Y_: mnist.test.labels}))
