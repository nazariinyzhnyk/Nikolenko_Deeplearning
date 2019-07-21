import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Activation

# Variable initialization example
w = tf.Variable(tf.random.normal([3, 2], mean=0.0, stddev=0.4), name='weigths')
b = tf.Variable(tf.zeros([2]), name='biases')
print(w)
print(b)

# Placeholder example
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
output = tf.multiply(x, y)

with tf.Session() as sess:
    res = sess.run(output, feed_dict={
        x: 2,
        y: 3
    })

print(res)

# Broadcasting example
m = tf.Variable(tf.random.normal([2, 10], mean=0.0, stddev=0.4), name='matrix')
v = tf.Variable(tf.random.normal([10], mean=0.0, stddev=0.4), name='vector')
init_op = tf.initialize_all_variables()
output = tf.add(m, v)

with tf.Session() as sess:
    sess.run(init_op)
    res = sess.run(output)

print(res)

# Reduction operations example

x = tf.placeholder(tf.float32, [2, 3])
output = tf.reduce_mean(x, axis=1)

with tf.Session() as sess:
    res = sess.run(output, feed_dict={
        x: [[1, 2, 3],  # 2
            [4, 5, 6]]  # 5
    })

print(res)  # [2. 5.]


# Function example

def linear_transform(vec, shape):
    w = tf.Variable(tf.random.normal(shape, mean=0.0, stddev=0.4), name='matrix')
    return tf.matmul(vec, w)


vec1 = tf.Variable(tf.random.normal([3, 3], mean=0.0, stddev=0.4), name='vector')
vec2 = tf.Variable(tf.random.normal([3, 3], mean=0.0, stddev=0.4), name='vector')
result1 = linear_transform(vec1, [3, 3])
result2 = linear_transform(vec2, [3, 3])
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    res1 = sess.run(result1)
    res2 = sess.run(result2)

print(res1)
print(res2)

# Linear regression example

num_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (num_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (num_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')
    b = tf.Variable(tf.zeros(1, ), name='bias')

y_pred = tf.matmul(X, k) + b
loss = tf.reduce_sum((y - y_pred) ** 2)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

display_step = 100
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(num_steps):
        idx = np.random.choice(num_samples, batch_size)
        X_batch, y_batch = X_data[idx], y_data[idx]
        _, loss_val, k_val, b_val = sess.run([opt, loss, k, b],
                                             feed_dict={
                                                 X: X_batch,
                                                 y: y_batch
                                             })
        if (i + 1) % display_step == 0:
            print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i+1, loss_val, k_val, b_val))


logr = Sequential()
logr.add(Dense(1, input_dim=2, activation='sigmoid'))
logr.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # sgd


def sampler(n, x, y):
    return np.random.normal(size=[n, 2]) + [x, y]


def sample_data(n=1000, p0=(-1., -1), p1=(1, 1)):
    zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
    labels = np.vstack([zeros, ones])
    z_sample = sampler(n, x=p0[0], y=p0[1])
    o_sample = sampler(n, x=p1[0], y=p1[1])
    return np.vstack([z_sample, o_sample]), labels


X_train, y_train = sample_data()
X_test, y_test = sample_data(100)

logr.fit(X_train, y_train, batch_size=16, epochs=100,
         verbose=1, validation_data=(X_test, y_test))
