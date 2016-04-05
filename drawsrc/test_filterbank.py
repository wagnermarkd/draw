from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mnist_learn import binarize
from filterbank import banks
import numpy as np

sess = tf.Session()
with sess.as_default():
    shape = (28,28)
    N = 5
    delta = [1., 1.]
    gamma = [2., 2.]
    sigma = [0.1, 2.]
    center_x = [2.,3.]
    center_y = [2.5,3.]
    FX, FY = banks(center_x,center_y,sigma,delta,gamma,N,shape)
    print FX.eval()
    print FY.eval()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, valid_data, test_data = binarize(mnist.train.images), binarize(mnist.validation.images), binarize(mnist.test.images)
    data = np.reshape(train_data[:2], [-1, 28,28])
    print tf.batch_matmul(FX, data).eval()



