import tensorflow as tf


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


vec1 = tf.Variable(tf.random.normal([3, 3], mean=0.0, stddev=0.4), name='vector')
vec2 = tf.Variable(tf.random.normal([3, 3], mean=0.0, stddev=0.4), name='vector')
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    with tf.variable_scope('linear_transformers') as scope:
        result1 = linear_transform(vec1, [3, 3])
        scope.reuse_variables()
        result2 = linear_transform(vec2, [3, 3])
        res1 = sess.run(result1)
        res2 = sess.run(result2)

print(res1)
print(res2)


# Linear regression example
