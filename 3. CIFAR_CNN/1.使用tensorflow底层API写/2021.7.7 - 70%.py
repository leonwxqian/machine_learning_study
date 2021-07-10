import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets

tf.disable_v2_behavior()

IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH  # 3072
RESIZE_SIZE = IMAGE_SIZE


def get_batch(image, label, batch_size, now_batch, total_batch):
    maximum_batch_no = total_batch // batch_size - 1

    now_batch = now_batch % (maximum_batch_no + 1)
    if now_batch == maximum_batch_no:
        now_batch = maximum_batch_no - 1

    if now_batch < total_batch - 1:
        image_batch = image[now_batch * batch_size: (now_batch + 1) * batch_size]
        label_batch = label[now_batch * batch_size: (now_batch + 1) * batch_size]
    else:
        image_batch = image[now_batch * batch_size:]
        label_batch = label[now_batch * batch_size:]
    return image_batch, label_batch


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = y_train.reshape(y_train.shape[0])
y_train_1d = tf.cast(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train_1d, 10)
 
y_test = y_test.reshape(y_test.shape[0])
y_test_1d = tf.cast(y_test, dtype=tf.int32)
y_test = tf.one_hot(y_test_1d, 10)
 
# 使用类似上次的CNN, CIFAR也有10类
x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

OUTPUT_SIZE = 96
INPUT_DEPTH = 3

CONV1_SIZE = 3
CONV2_SIZE = 5

MAX_POOL1_SIZE = 2
MAX_POOL2_SIZE = 2

x_image = tf.reshape(x, [-1, 32, 32, 3])

w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))
h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') , b_conv1))
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2))
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# full connecting layer 1
w_fc1 = tf.Variable(tf.truncated_normal([4096, 384], stddev=0.04))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 4096])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


w_fc2 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.04))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[192]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

w_fc3 = tf.Variable(tf.truncated_normal([192, 10], stddev=1/192.0))
b_fc3 = tf.Variable(tf.constant(0.0, shape=[10]))
y_conv = tf.add(tf.matmul(h_fc2, w_fc3), b_fc3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate=1E-4 ).minimize(cross_entropy)

BATCH_SIZE = 100
EPOCHES = 2000

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE) 


for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, 200, i, 50000)
    xb = x_batch
    yb = y_batch.eval(session=sess)

    (x_small_test_batch, y_small_test_batch) = get_batch(x_test, y_test, 500, i, 10000)
    xt = x_small_test_batch
    yt = y_small_test_batch.eval(session=sess)

    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt})
        print("step %d, acc: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: xb, y_: yb})

train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt})
print("step final, acc: %g" % train_accuracy)
