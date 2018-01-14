import numpy as np
import tensorflow as tf
from tensorflow.python import keras as ks
from Potato_Peal import identity_block, convolutional_block

def identity_block_test():
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a', seed=0)
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, ks.backend.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))
        # with seed=0 on the initialize of the glorot_uniform in the identity block the result should be
        # [ 0.94822985 0. 1.16101444 2.747859 0. 1.36677003]


def convolutional_block_test():
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a', seed=0)
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, ks.backend.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))

convolutional_block_test()