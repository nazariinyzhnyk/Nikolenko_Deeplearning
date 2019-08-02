import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

batch_size, latent_space, learning_rate = 64, 128, 0.1

ae_weights = {"encoder_w": tf.Variable(tf.truncated_normal([784, latent_space], stddev=0.1)),
              "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
              "decoder_w": tf.Variable(tf.truncated_normal([latent_space, 784], stddev=0.1)),
              "decoder_b": tf.Variable(tf.truncated_normal([784], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, 784])
hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
visible_logits = tf.matmul(hidden, ae_weights["decoder_w"] + ae_weights["decoder_b"])
visible = tf.nn.sigmoid(visible_logits)

ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=ae_input))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

rho = 0.05
beta = 1.0
data_rho = tf.reduce_mean(hidden, 0)
reg_cost = -tf.reduce_mean(tf.log(data_rho / rho) * rho +
                           tf.log((1 - data_rho) / (1 - rho)) * (1 - rho))

total_cost = ae_cost + beta * reg_cost

ae_op = optimizer.minimize(total_cost)  # ae_cost

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):  # 10000
        x_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(ae_op, feed_dict={ae_input: x_batch})




