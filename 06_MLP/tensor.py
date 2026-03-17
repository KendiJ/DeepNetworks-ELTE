import tensor as tf

w1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
w2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
b2 = tf.Variable(tf.random_normal([n_output]))
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
layer_1 = tf.sigmoid(tf.add(tf.matmul(X, w1), b1))
layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
loss_op = tf.reduce_mean(tf.square(tf.subtract(layer_2, y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)