import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers import ZeroPadding2D, LayerNormalization
from keras.layers.merge import concatenate
from keras.models import Model

import nnlayers

tf.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# label_mode: one of "fine", "coarse". If it is "fine" the category labels are the fine-grained labels,
# if it is "coarse" the output labels are the coarse-grained superclasses. we've used fine here, so it is divided
# into 100 categories.
# https://keras.io/api/datasets/cifar100/
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

#input = Input(shape=(32, 32, 3))

IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH  # 3072

x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, 32, 32, 3])


def inception(input, f1, f2, f3, f4 ,f5 ,f6):
    if(f1 != 0):
        inception_1x1 = nnlayers.conv_2d(input, filter_count=f1, kernel_size=1, strides=1)
    inception_3x3_reduce = nnlayers.conv_2d(input, filter_count=f2, kernel_size=1, strides=1)
    inception_3x3 = nnlayers.conv_2d(inception_3x3_reduce, filter_count=f3, kernel_size=3, strides=1)
    inception_5x5_reduce = nnlayers.conv_2d(input, filter_count=f4, kernel_size=1, strides=1)

    inception_3x3a = nnlayers.conv_2d(inception_5x5_reduce, filter_count=f5, kernel_size=3, strides=1)
    inception_3x3b = nnlayers.conv_2d(inception_3x3a, filter_count=f5, kernel_size=3, strides=1)

    if(f6 == 0):
        inception_pool = nnlayers.max_pooling(input, 3, 1)
    else:
        inception_pool = nnlayers.avg_pooling(input, 3, 1)
        inception_pool_proj = nnlayers.conv_2d(inception_pool, filter_count=f6, kernel_size=1, strides=1)

    if(f1 == 0 and f6 == 0):
        output = tf.concat([inception_3x3, inception_3x3b, inception_pool],axis=3)
    elif (f1 == 0):
        output = tf.concat([inception_3x3, inception_3x3b, inception_pool_proj], axis=3)
    else:
        output = tf.concat([inception_1x1, inception_3x3, inception_3x3b, inception_pool], axis=3)
    return output

def build_model():
    conv1_7x7_s2 = nnlayers.conv_2d(x_image, filter_count=64, kernel_size=7, strides=1) #2)
    #pool1_3x3_s2 = max_pool('pool1_3x3_s2', conv1_7x7_s2, 3, 2)
    #pool1_norm1 = lrn('pool1_norm1', pool1_3x3_s2)
    conv2_3x3_reduce = nnlayers.conv_2d(conv1_7x7_s2, filter_count=64, kernel_size=1, strides=1)
    conv2_3x3 = nnlayers.conv_2d(conv2_3x3_reduce, filter_count=192, kernel_size=3, strides=1)
    conv2_norm2 = nnlayers.lrn(conv2_3x3)
    pool2_3x3_s2 = nnlayers.max_pooling(conv2_norm2, 3, 2)

    inception_3a_output = inception(pool2_3x3_s2, 64, 64, 64, 64, 96, 32)

    inception_3b_output = inception(inception_3a_output, 64, 64, 96, 64, 96, 64)

    inception_3c_output = inception(inception_3b_output, 0, 128, 160, 64, 96, 0)

    pool3_3x3_s2 = nnlayers.max_pooling(inception_3c_output, 3, 2)

    inception_4a_output = inception(pool3_3x3_s2, 224, 64, 96, 96, 128, 128)

    inception_4b_output = inception(inception_4a_output, 192, 96, 128, 96, 128, 128)

    inception_4c_output = inception(inception_4b_output, 160, 128, 160, 128, 160, 128)

    inception_4d_output = inception(inception_4c_output, 96, 128, 192, 160, 192, 128)

    inception_4e_output = inception(inception_4d_output, 0, 128, 192, 192, 256, 0)

    pool4_3x3_s2 = nnlayers.max_pooling(inception_4e_output, 3, 2)

    inception_5a_output = inception(pool4_3x3_s2, 352, 192, 320, 160, 224, 128)

    inception_5b_output = inception(inception_5a_output, 352, 192, 320, 192, 224, 128)

    pool5_7x7_s1 = nnlayers.avg_pooling(inception_5b_output, 7, 1)
    pool5_drop_7x7_s1 = nnlayers.dropout(pool5_7x7_s1, 0.6)

    #logits = fc('loss3_classifier', pool5_drop_7x7_s1, out_nodes=100)
    logits = nnlayers.dense(pool5_drop_7x7_s1, 100, kernel_regularizer='none', activation='relu')
    return logits

k = build_model()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=k))
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.002).minimize(cross_entropy)

#x_train = x_train.astype("float32")
#x_test = x_test.astype("float32")
#x_train /= 255.0
#x_test /= 255.0

sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(k, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE)
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE)

BATCH_SIZE = 64
TRAINING_DATASET_COUNT = 50000
# train 100 rounds on every dataset in training set.
EPOCHES = 100
BATCHES = (TRAINING_DATASET_COUNT // BATCH_SIZE) * EPOCHES

(x_small_test_batch, y_small_test_batch) = nnlayers.get_batch(x_test, y_test, 100, 1, 10000)

with tf.Session() as tfsess:
    tf.global_variables_initializer().run()
    for i in range(BATCHES):
        (x_batch, y_batch) = nnlayers.get_batch(x_train, y_train, BATCH_SIZE, i, 50000)
        _, loss_value = tfsess.run([train_step, cross_entropy], feed_dict={x: x_batch, y_: y_batch})
        #report loss every 10 batches
        if i % 10 == 0:
            print("batch %d, loss: %g" % (i, loss_value))

        #test after a whole epoch
        if i > 0 and i % (TRAINING_DATASET_COUNT // BATCH_SIZE) == 0:
            train_accuracy = accuracy.eval(feed_dict={x: x_small_test_batch, y_: y_small_test_batch})
            print("batch %d, loss: %g, acc: %g" % (i, loss_value, train_accuracy))

    train_accuracy = accuracy.eval(feed_dict={x: x_train, y_: y_test})
    print("batch final, acc: %g" % train_accuracy)