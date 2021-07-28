import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets
import nnlayers
import tfimg

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

    if now_batch < maximum_batch_no - 1:
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

x_image = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = nnlayers.conv_2d(x_image, 3, 32)
h_conv2 = nnlayers.conv_2d(h_conv1, 3, 32)
h_pool1 = nnlayers.max_pooling(h_conv2, 2, 2, padding='SAME')
h_dropout1 = nnlayers.dropout(h_pool1, 0.25)
h_conv3 = nnlayers.conv_2d(h_dropout1, 3, 64)
h_conv4 = nnlayers.conv_2d(h_conv3, 3, 64)
h_pool2 = nnlayers.max_pooling(h_conv4, 2, 2, padding='SAME')
h_dropout2 = nnlayers.dropout(h_pool2, 0.25)
h_flatten = nnlayers.flatten(h_dropout2)
h_dense1 = nnlayers.dense(h_flatten, 512, kernel_regularizer='none')
#h_dropout3 = nnlayers.dropout(h_dense1, 0.25)
y_conv = (nnlayers.dense(h_dense1, 10, kernel_regularizer='none', stddev=1/192.0, bias=0.0))
print(np.shape(y_conv))


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)

BATCH_SIZE = 500
EPOCHES = 8000

sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE)

#these steps lowers the accuracy:
#x_train = x_train.astype("float32")
#x_test = x_test.astype("float32")
#x_train = x_train / 255.0
#x_test = x_test / 255.0

y_train = y_train.eval(session=sess)
y_test = y_test.eval(session=sess)

#tf.get_default_graph().finalize()

(xt, yt) = get_batch(x_test, y_test, 1000, 1, 10000)
with tf.Session() as tfsess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHES):
        (xb, yb) = get_batch(x_train, y_train, BATCH_SIZE, i, 50000)
        _, loss_value = tfsess.run([train_step, cross_entropy], feed_dict={x: xb, y_: yb})
        #report loss every 10 batches
        if i % 10 == 0:
            print("batch %d, loss: %g" % (i, loss_value))

        if i > 0 and i % 100 == 0:  # 500x100 = 1 epoch
            test_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt})
            train_accuracy = accuracy.eval(feed_dict={x: xb, y_: yb})
            print("batch %d, test acc: %g, train acc: %g" % (i, test_accuracy, train_accuracy))

train_accuracy = accuracy.eval(feed_dict={x: xt, y_: yt})
print("step final, acc: %g" % train_accuracy)
