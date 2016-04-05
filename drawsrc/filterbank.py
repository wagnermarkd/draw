import tensorflow as tf
import math

def batched_dot(A, B):
    return 0

def banks(center_x, center_y, sigma, delta, N, shape):
    tol = 1e-5

    (height, width) = shape
    # TODO: lasagne-draw has a bug here that was fixed upstream. Should patch for them
    stride = tf.expand_dims(delta,dim=1)*(tf.cast(tf.range(N), tf.float32) - N/2. + 0.5)
    stride = tf.Print(stride, [tf.shape(stride)])
    mu_x = tf.add(tf.expand_dims(center_x, dim=1), stride)
    mu_y = tf.add(tf.expand_dims(center_y, dim=1), stride)

    a = tf.cast(tf.range(width), tf.float32)
    b = tf.cast(tf.range(height), tf.float32)

    FX = tf.exp( -(a - tf.expand_dims(mu_x, 2))**2 / 2. / tf.expand_dims(tf.expand_dims(sigma, 1), 2)**2)
    FY = tf.exp( -(b - tf.expand_dims(mu_y, 2))**2 / 2. / tf.expand_dims(tf.expand_dims(sigma, 1), 2)**2)
    FX = FX / (tf.expand_dims(tf.reduce_sum(FX, reduction_indices=1),1) + tol)
    FY = FY / (tf.expand_dims(tf.reduce_sum(FY, reduction_indices=1),1) + tol)

    return FX, FY
    print FY.eval()
    print FX.eval()
    print mu_y.eval()

def read(images, N, delta, gamma, sigma, center_x, center_y):
    #TODO: Make configurable shape
    FX, FY = banks(center_x, center_y, sigma, delta, gamma, N, (28,28))

    I = tf.reshape(images, [-1, 28, 28])

    I = tf.batch_matmul(I, FX);
    I = tf.batch_matmul(tf.transpose(FY, [2,1]), I)
    return gamma*tf.reshape(I, [-1, N*N])

# NOTE: These are DIFFERENT centers, gammas, etc. That's not the case in other implementations
def write(windows, N, center_x, center_y, delta, sigma):
    W = tf.reshape(windows, [-1, N, N])
    FX, FY = banks(center_x, center_y, sigma, delta, N, (28,28))

    I = tf.batch_matmul(W, FY);
    I = tf.batch_matmul(tf.transpose(FX, [2,1]), I)

    return (1/gamma)*tf.reshape(I, [-1, N*N])

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
