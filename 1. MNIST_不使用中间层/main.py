import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


def get_batch(image, label, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        image_batch = image[now_batch * batch_size: (now_batch + 1) * batch_size]
        label_batch = label[now_batch * batch_size: (now_batch + 1) * batch_size]
    else:
        image_batch = image[now_batch * batch_size:]
        label_batch = label[now_batch * batch_size:]
    return image_batch, label_batch


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


(x_train, y_train_1d), (x_test, y_test_1d) = load_data(path="./mnist.npz")

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# the load_data gives us a (60000, ) y.
# we need to set it to (60000, 10)
y_train_1d = tf.cast(y_train_1d, dtype=tf.int32)
y_train = tf.one_hot(y_train_1d, 10)

y_test_1d = tf.cast(y_test_1d, dtype=tf.int32)
y_test = tf.one_hot(y_test_1d, 10)
print(y_test[0])
################################################
#  Experiment:
#  No layer between input and output
#
#  X(input) SOFTMAX  Y(output)
#   784x1   --w---    10x1
#            \b/
#
################################################


# Each picture in mnist dataset has (? pictures) x 784 (28x28) pixels.
# [None = ? , 784 = 28x28]
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
# And it have 10 different labels, set output shape to 10
# Also, because we don't know how many pictures will be given here,
# so leave the first item of shape as None.
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# Input 784 output 10
w = tf.Variable(tf.zeros([784, 10]))
# b is added to the output, so it should have the same shape as y
b = tf.Variable(tf.zeros([10]))
# y = softmax( w * x + b ), our predicted value
y = tf.nn.softmax(tf.matmul(x, w) + b)

BATCH_SIZE = 100
EPOCHES = 300
lr = 0.5

# Initialize loss function using cross entropy.
#
# H (y) = - Σ y' log 1/y
#  y'       i  i        i
#
# reduction_indices: the desired dimension of data being processed.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

# Set learning rate, and loss function cross entropy
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, 200, i, EPOCHES)
    #print(type(x_batch), type(y_batch))
    # x: <class 'numpy.ndarray'>  y : <class 'tensorflow.python.framework.ops.Tensor'>
    # so y needs to be converted to numpy.ndarray to satisfy sess.run ..
    xb = x_batch
    yb = y_batch.eval(session=sess)
    sess.run(train_step, feed_dict={x: xb, y_: yb})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    xt = x_test
    yt = y_test.eval(session=sess)
    test_data = {x: xt, y_: yt}
    print('epoches %d, and its accuracy is: %f' % (i, accuracy.eval(test_data)))
