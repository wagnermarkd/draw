import tensorflow as tf
import math

def batched_dot(A, B):
    return 0

def banks(center_x, center_y, sigma, delta, N, shape):
    tol = 1e-5

    (height, width) = shape
    # TODO: lasagne-draw has a bug here that was fixed upstream. Should patch for them
    stride = tf.expand_dims(delta,dim=1)*(tf.cast(tf.range(N), tf.float32) - N/2. + 0.5)
    mu_x = tf.add(tf.expand_dims(center_x, dim=1), stride)
    mu_y = tf.add(tf.expand_dims(center_y, dim=1), stride)

    a = tf.cast(tf.range(width), tf.float32)
    b = tf.cast(tf.range(height), tf.float32)

    FX = tf.exp( -(a - tf.expand_dims(mu_x, 2))**2 / 2. / tf.expand_dims(tf.expand_dims(sigma + tol, 1), 2)**2)
    FY = tf.exp( -(b - tf.expand_dims(mu_y, 2))**2 / 2. / tf.expand_dims(tf.expand_dims(sigma + tol, 1), 2)**2)
    FX = FX / (tf.expand_dims(tf.reduce_sum(FX, reduction_indices=1),1) + tol)
    FY = FY / (tf.expand_dims(tf.reduce_sum(FY, reduction_indices=1),1) + tol)

    return FX, FY
    print FY.eval()
    print FX.eval()
    print mu_y.eval()

def read(images, N, delta, gamma, sigma, center_x, center_y):
    #TODO: Make configurable shape
    FX, FY = banks(center_x, center_y, sigma, delta, N, (28,28))

    I = tf.reshape(images, [-1, 28, 28])

    I = tf.batch_matmul(FY, I);
    I = tf.batch_matmul(I, tf.transpose(FX, [0,2,1]))

    return tf.expand_dims(gamma,1)*tf.reshape(I, [-1, N*N])

def get_params(vector, shape, strides):
    values = tf.unpack(tf.transpose(vector))
    center_x = ((shape[0] + 1)/2)*(values[0] + 1)
    center_y = ((shape[1] + 1)/2)*(values[1] + 1)
    sigma = tf.sqrt(tf.exp(values[2]))
    delta = ((max(shape) - 1)/(strides-1))*tf.exp(values[3])
    gamma = tf.exp(values[4])

    return center_x, center_y, sigma, delta, gamma

def read_vec(images, vector, shape, strides):
    center_x, center_y, sigma, delta, gamma = get_params(vector, shape, strides)

    return read(images, strides, delta, gamma, sigma, center_x, center_y)

# NOTE: These are DIFFERENT centers, gammas, etc. That's not the case in other implementations
def write(windows, N, center_x, center_y, delta, sigma, gamma):
    tol = 1e-5
    W = tf.reshape(windows, [-1, N, N])
    FX, FY = banks(center_x, center_y, sigma, delta, N, (28,28))

    I = tf.batch_matmul(W, FY);
    I = tf.batch_matmul(tf.transpose(FX, [0,2,1]), I)

    return tf.expand_dims(1/(gamma + tol),1)*tf.reshape(I, [-1, 28*28])

def write_vec(windows, vector, shape, strides):
    center_x, center_y, sigma, delta, gamma = get_params(vector, shape, strides)
    return write(windows, strides, center_x, center_y, delta, sigma, gamma)

if __name__ == "__main__":
    sess = tf.Session()
    with sess.as_default():
        shape = (28,28)
        N = 3
        delta = [2., 2.]
        gamma = [2., 2.]
        sigma = [0.1, 0.2]
        center_x = [2.,3.]
        center_y = [2.5,3.]

        banks(center_x,center_y,sigma,delta,gamma,N,shape)
    sess.close()
