import tensorflow as tf
import numpy as np


# load datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# using 10000 samples as train data, 200 samples as test data
X_train, Y_train = mnist.train.next_batch(50000)
X_test, Y_test = mnist.train.next_batch(200)


xtr = tf.placeholder(tf.float32, [None, 784])
xts = tf.placeholder(tf.float32, [784])

# using L1 to calculate distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xts))), reduction_indices=1)
# find min distance index from distance tensor
pred = tf.argmin(distance, 0)

accuracy = 0.

init = tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):
        nn_index = sess.run(pred, feed_dict={xtr: X_train, xts: X_test[i, :]})
        print("Test", i, "Prediction:", np.argmax(Y_train[nn_index]),
              "True Class", np.argmax(Y_test[i]))
        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print("Accuracy:", accuracy)