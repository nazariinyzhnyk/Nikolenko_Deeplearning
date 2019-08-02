import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

batch_size, learning_rate = 64, 0.01

ae_weights = {
    "conv": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    "b_hidden": tf.Variable(tf.truncated_normal([4], stddev=0.1)),
    "deconv": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    "b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1))}

input_shape = tf.stack([batch_size, 28, 28, 1])
ae_input = tf.placeholder(tf.float32, [batch_size, 784])
images = tf.reshape(ae_input, [-1, 28, 28, 1])
hidden_logits = tf.nn.conv2d(images, ae_weights["conv"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_hidden"]

hidden = tf.nn.sigmoid(hidden_logits)
visible_logits = tf.nn.conv2d_transpose(hidden, ae_weights["deconv"], input_shape, strides=[1, 2, 2, 1], padding="SAME") \
                 + ae_weights["b_visible"]

visible = tf.nn.sigmoid(visible_logits)

optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=images))
conv_op = optimizer.minimize(conv_cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):  # 10000
        x_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(conv_op, feed_dict={ae_input: x_batch})
