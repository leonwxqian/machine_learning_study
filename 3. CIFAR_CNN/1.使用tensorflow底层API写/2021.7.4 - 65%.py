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
    now_batch = (now_batch) % (total_batch // batch_size  + 1)
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

# ytest, before shape= (10000, 1) --> (10000)
y_test = y_test.reshape(y_test.shape[0])
y_test_1d = tf.cast(y_test, dtype=tf.int32)
y_test = tf.one_hot(y_test_1d, 10)

print("ytest, after onehot, shape=", np.shape(y_test))
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
# conv 1, the cores are required to be 32 x groups(3).. so use 96 here.
w_conv1 = tf.Variable(tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, INPUT_DEPTH, OUTPUT_SIZE], stddev=0.01))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
print(np.shape(h_conv1))
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, MAX_POOL1_SIZE, MAX_POOL1_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print(np.shape(h_pool1))

w_conv2 = tf.Variable(tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, INPUT_DEPTH, OUTPUT_SIZE * 2], stddev=0.01))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE * 2]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
print(np.shape(h_conv2))
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, MAX_POOL2_SIZE, MAX_POOL2_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print(np.shape(h_pool2))

FC1_OUT_COUNT = 1024
FC2_OUT_COUNT = 256

# full connecting layer 1
w_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * OUTPUT_SIZE * 2, FC1_OUT_COUNT], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.01, shape=[FC1_OUT_COUNT]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * OUTPUT_SIZE * 2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout, 0.5 when training, 1 when testing
keep_prob = tf.compat.v1.placeholder(tf.float32)
# the tensorflow asks me to use rate=1-keep_prob instead, okay then:
h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)

# full connecting layer 2, convert h_fc1_drop to scores for 10 classes
w_fc2 = tf.Variable(tf.truncated_normal([FC1_OUT_COUNT, FC2_OUT_COUNT], stddev=0.01))
b_fc2 = tf.Variable(tf.constant(0.01, shape=[FC2_OUT_COUNT]))
h_pool_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 2 layers -- acc = 0.449
# add full connecting layer 3.
w_fc3 = tf.Variable(tf.truncated_normal([FC2_OUT_COUNT, 10], stddev=0.01))
b_fc3 = tf.Variable(tf.constant(0.01, shape=[10]))
h_pool_fc3 = tf.reshape(h_pool_fc2, [-1, FC2_OUT_COUNT])
y_conv = tf.nn.softmax(tf.matmul(h_pool_fc3, w_fc3) + b_fc3 )

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate=1E-4, beta1=0.99,
                                    beta2=0.999,
                                    epsilon=1e-08 ).minimize(cross_entropy)

BATCH_SIZE = 100
EPOCHES = 250

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# should i use "flatten" instead???
print(np.shape(x_train))
# shape of x_train:  (32, 32, 3)
# we'd like it to be  10000, image_size
x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE)
print("1")
# xt = x_test
# print("xt shape" , np.shape(xt))
# yt = y_test.eval(session=sess)
# print("yt shape" , np.shape(yt))  #it requires a [?, 3072] shape

(x_small_test_batch, y_small_test_batch) = get_batch(x_test, y_test, 1000, 1, 10000)
xt = x_small_test_batch
yt = y_small_test_batch.eval(session=sess)

for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, 200, i, 50000)
    xb = x_batch
    yb = y_batch.eval(session=sess)

    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt, keep_prob: 1.0})
        print("step %d, acc: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: xb, y_: yb, keep_prob: 0.5})
    train_step.run(feed_dict={x: xb, y_: yb, keep_prob: 0.5})

train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt, keep_prob: 1.0})
print("step final, acc: %g" % train_accuracy)
