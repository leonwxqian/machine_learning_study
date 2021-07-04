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

# the load_data gives us a (60000, ) y.
# we need to set it to (60000, 10)
y_train_1d = tf.cast(y_train_1d, dtype=tf.int32)
y_train = tf.one_hot(y_train_1d, 10)

y_test_1d = tf.cast(y_test_1d, dtype=tf.int32)
y_test = tf.one_hot(y_test_1d, 10)
################################################
#  Experiment:
#  2 layers between input and output
#   X - layer1 - layer2 - y
################################################

# X: input, y_: given image labels.
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
# conv 1
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
print(np.shape(h_conv1))
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(np.shape(h_pool1))

w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
print(np.shape(h_conv2))
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(np.shape(h_pool2))
#full connecting layer
w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
print(np.shape(h_pool2_flat))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
print(np.shape(h_fc1))
#dropout, 0.5 when training, 1 when testing
keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#full connecting layer 2, convert h_fc1_drop to scores for 10 classes
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
print(np.shape(y_conv))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


BATCH_SIZE = 100
EPOCHES = 600

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
xt = x_test
yt = y_test.eval(session=sess)

for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, 200, i, EPOCHES)
    xb = x_batch
    yb = y_batch.eval(session=sess)

    if i % 20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt, keep_prob: 1.0})
        print("step %d, acc: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: xb, y_: yb, keep_prob: 0.5})

train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt, keep_prob: 1.0})
print("step final, acc: %g" % (train_accuracy))