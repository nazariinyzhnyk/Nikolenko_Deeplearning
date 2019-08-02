import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

batch_size, learning_rate = 64, 0.01

ae_weights = {
    "conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    "b_conv1": tf.Variable(tf.truncated_normal([4], stddev=0.1)),
    "conv2": tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
    "b_hidden": tf.Variable(tf.truncated_normal([16], stddev=0.1)),
    "deconv1": tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
    "b_deconv": tf.Variable(tf.truncated_normal([1], stddev=0.1)),
    "deconv2": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    "b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1))}

input_shape = tf.stack([batch_size, 28, 28, 1])
ae_input = tf.placeholder(tf.float32, [batch_size, 784])
images = tf.reshape(ae_input, [-1, 28, 28, 1])
h1_shape = tf.stack([batch_size, 14, 14, 4])

conv_h1_logits = tf.nn.conv2d(images, ae_weights["conv1"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_conv1"]
conv_h1 = tf.nn.relu(conv_h1_logits)

hidden_logits = tf.nn.conv2d(conv_h1, ae_weights["conv2"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_hidden"]
hidden = tf.nn.relu(hidden_logits)

deconv_h1_logits = tf.nn.conv2d_transpose(hidden, ae_weights["deconv1"], h1_shape,
                                          strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_deconv"]
deconv_h1 = tf.nn.relu(deconv_h1_logits)

visible_logits = tf.nn.conv2d_transpose(deconv_h1, ae_weights["deconv2"], input_shape,
                                        strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_visible"]

visible = tf.nn.sigmoid(visible_logits)

optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible, labels=images))
conv_op = optimizer.minimize(conv_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):  # 10000
        x_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(conv_op, feed_dict={ae_input: x_batch})
