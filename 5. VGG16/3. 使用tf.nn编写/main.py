import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers

tf.disable_v2_behavior()

# label_mode: one of "fine", "coarse". If it is "fine" the category labels are the fine-grained labels,
# if it is "coarse" the output labels are the coarse-grained superclasses. we've used fine here, so it is divided
# into 100 categories.
# https://keras.io/api/datasets/cifar100/
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

# one hot
y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)


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


def conv_2d(input, kernel_size, filter_count, strides=1, kernel_regularizer='none'):
    print(input.shape[-1])
    w_conv = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input.shape[-1], filter_count], stddev=0.01))
    b_conv = tf.Variable(tf.constant(0.1, shape=[filter_count]))

    if kernel_regularizer == 'none':
        return tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv)
    elif kernel_regularizer == 'l2':
        return tf.nn.l2_normalize(tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv), dim=0, epsilon=0.0005)


def max_pooling(input, kernel_size, strides=0):
    if strides == 0:
        strides = kernel_size
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1],
                          padding="VALID")

#use tf.nn.xw_plus_b
def dense(input, units, stddev=0.04, kernel_regularizer='none', activation='relu'):
    w_fc = tf.Variable(
        tf.truncated_normal([input.shape[-1], units], stddev=stddev))  # kernel initializer=truncated_normal
    bias_init = 0.01
    b_fc = tf.Variable(tf.constant(bias_init, shape=[units]))  # bias_initializer=constant 0.01
    logits = tf.nn.xw_plus_b(x=input, weights=w_fc, biases=b_fc)

    if kernel_regularizer == 'none':
        return tf.nn.relu(logits)
    elif kernel_regularizer == 'l2' and activation == 'relu':
        return tf.nn.l2_normalize(tf.nn.relu(logits), dim=0, epsilon=0.0005)
    elif kernel_regularizer == 'none' and activation == 'softmax':
        return tf.nn.softmax(logits)


def batch_normalization(inputs):
    mean, variance = tf.nn.moments(inputs, axes=0)
    return tf.nn.batch_normalization(
        inputs, mean, variance, None, None, 1e-12, name=None  # try 0.001 also
    )


def dropout(input, percentage):
    return tf.nn.dropout(input, rate=1 - percentage)


def flatten(input, flat_to):
    return tf.reshape(input, [-1, flat_to])


IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH  # 3072

x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = conv_2d(x_image, 3, 64)
h_norm1 = batch_normalization(h_conv1)
h_dropout1 = dropout(h_norm1, 0.3)
print("layer1, ", np.shape(h_dropout1))

h_conv2 = conv_2d(h_dropout1, 3, 64, kernel_regularizer='l2')
h_norm2 = batch_normalization(h_conv2)
# according to https://keras.io/zh/layers/pooling/, strides is equal to pool_size by default.
h_pool2 = max_pooling(h_norm2, 2, 2)
print("layer2, ", np.shape(h_pool2))

h_conv3 = conv_2d(h_pool2, 3, 128, kernel_regularizer='l2')
h_norm3 = batch_normalization(h_conv3)
h_dropout3 = dropout(h_norm3, 0.4)
print("layer3, ", np.shape(h_dropout3))

# should be layer#4... i made a silly mistake by not changing its index when writing this..
h_conv3_b = conv_2d(h_dropout3, 3, 128, kernel_regularizer='l2')
h_norm3_b = batch_normalization(h_conv3_b)
h_pool3_b = max_pooling(h_norm3_b, 2, 2)
print("layer4, ", np.shape(h_pool3_b))

h_conv4 = conv_2d(h_pool3_b, 3, 256, kernel_regularizer='l2')
h_norm4 = batch_normalization(h_conv4)
h_dropout4 = dropout(h_norm4, 0.4)

h_conv5 = conv_2d(h_dropout4, 3, 256, kernel_regularizer='l2')
h_norm5 = batch_normalization(h_conv5)
h_dropout5 = dropout(h_norm5, 0.4)

h_conv6 = conv_2d(h_dropout5, 3, 256, kernel_regularizer='l2')
h_norm6 = batch_normalization(h_conv6)
h_pool6 = max_pooling(h_norm6, 2, 2)

h_conv7 = conv_2d(h_pool6, 3, 512, kernel_regularizer='l2')
h_norm7 = batch_normalization(h_conv7)
h_dropout7 = dropout(h_norm7, 0.4)

h_conv8 = conv_2d(h_dropout7, 3, 512, kernel_regularizer='l2')
h_norm8 = batch_normalization(h_conv8)
h_dropout8 = dropout(h_norm8, 0.4)

h_conv9 = conv_2d(h_dropout8, 3, 512, kernel_regularizer='l2')
h_norm9 = batch_normalization(h_conv9)
h_pool9 = max_pooling(h_norm9, 2, 2)

h_conv10 = conv_2d(h_pool9, 3, 512, kernel_regularizer='l2')
h_norm10 = batch_normalization(h_conv10)
h_dropout10 = dropout(h_norm10, 0.4)

h_conv11 = conv_2d(h_dropout10, 3, 512, kernel_regularizer='l2')
h_norm11 = batch_normalization(h_conv11)
h_dropout11 = dropout(h_norm11, 0.4)

h_conv12 = conv_2d(h_dropout11, 3, 512, kernel_regularizer='l2')
h_norm12 = batch_normalization(h_conv12)
h_pool12 = max_pooling(h_norm12, 2, 2)
h_dropout12 = dropout(h_pool12, 0.5)
print(np.shape(h_dropout12))
h_flatten = flatten(h_dropout12, 512)

h_dense1 = dense(h_flatten, 512, kernel_regularizer='l2')
h_norm_d1 = batch_normalization(h_dense1)
h_dropout_d1 = dropout(h_norm_d1, 0.5)

# h_dense2 = dense(h_dropout_d1, 100, kernel_regularizer='none', activation='softmax')
h_dense2 = tf.layers.dense(inputs=h_dropout_d1, units=100)
print(np.shape(h_dense2))
logits = tf.nn.softmax(h_dense2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_dense2))
train_step = tf.compat.v1.train.MomentumOptimizer(0.03, 0.9).minimize(cross_entropy)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE)

BATCH_SIZE = 64
TRAINING_DATASET_COUNT = 50000
# train 100 rounds on every dataset in training set.
EPOCHES = (TRAINING_DATASET_COUNT // BATCH_SIZE) * 100

(x_small_test_batch, y_small_test_batch) = get_batch(x_test, y_test, 1000, 1, 10000)
for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, BATCH_SIZE, i, 50000)
    if i > 0 and i % (TRAINING_DATASET_COUNT // BATCH_SIZE) == 0:
        train_accuracy = accuracy.eval(feed_dict={x: x_small_test_batch, y_: y_small_test_batch})
        print("step %d, acc: %g" % (i, train_accuracy))
    # print(np.shape(y_batch))
    train_step.run(feed_dict={x: x_batch, y_: y_batch})

train_accuracy = accuracy.eval(feed_dict={x: x_train, y_: y_test})
print("step final, acc: %g" % train_accuracy)
