import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name='x_placeholder')


W_relu = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b_relu = tf.Variable(tf.truncated_normal([100], stddev=0.1))

h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)
keep_probability = tf.placeholder(tf.float32)

h_drop = tf.nn.dropout(h, keep_prob=keep_probability)

W = tf.Variable(tf.zeros([100, 10]))
b = tf.Variable(tf.zeros([10]))

logit = tf.matmul(h_drop, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_))
y = tf.nn.softmax(tf.matmul(h_drop, W) + b)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y_: batch_ys,
            keep_probability: 0.5
        })

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy: %s' %
          sess.run(accuracy, feed_dict={
              x: mnist.test.images,
              y_: mnist.test.labels,
              keep_probability: 1
          }))
