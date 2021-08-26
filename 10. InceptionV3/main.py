import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers import ZeroPadding2D, LayerNormalization
from tensorflow.keras.models import Model
import cv2
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


IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH

#_x_train = []
#_x_test = []
#for x_tr in x_train:
#    _x_train.append(cv2.resize(x_tr, (IMAGE_WIDTH, IMAGE_WIDTH)))
#for x_te in x_test:
#    _x_test.append(cv2.resize(x_te, (IMAGE_WIDTH, IMAGE_WIDTH)))
#print(np.array(_x_train).shape)
#_x_train = np.array(_x_train).reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, 3)
#_x_test = np.array(_x_test).reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, 3)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_WIDTH, 3])


def inception_v3(input, f1, f1_w, f2, f2_w, f3, f3_w, f4, f4_w, f5, f5_w, f6, f6_w, f7, f7_w, f8, f8_w, f3_h=8888, f4_h=8888):
    if(f3_h == 8888):
        f3_h = f3_w
    if(f4_h == 8888):
        f4_h = f4_w

    branch1x1 = nnlayers.conv2d_bn(input, f1, f1_w, f1_w)

    branch5x5 = nnlayers.conv2d_bn(input, f2, f2_w, f2_w)
    branch5x5 = nnlayers.conv2d_bn(branch5x5, f3, f3_w, f3_h)

    branch3x3dbl = nnlayers.conv2d_bn(input, f4, f4_w, f4_h)
    branch3x3dbl = nnlayers.conv2d_bn(branch3x3dbl, f5, f5_w, f5_w)
    branch3x3dbl = nnlayers.conv2d_bn(branch3x3dbl, f6, f6_w, f6_w)
    branch_pool = nnlayers.avg_pooling(branch3x3dbl, f7, f7_w)
    branch_pool = nnlayers.conv2d_bn(branch_pool, f8, f8_w, f8_w) 
    return tf.concat(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3)

def inception_v3_b(input, f1, f1_w, f2, f2_w, f3, f3_w, f3_h, f4, f4_w, f4_h, f5, f5_w, f6, f6_w, f6_h, f7, f7_w, f7_h, f8, f8_w, f8_h, f9, f9_w, f9_h, f10, f10_w):
    branch1x1 = nnlayers.conv2d_bn(input, f1, f1_w, f1_w)

    branch7x7 = nnlayers.conv2d_bn(input, f2, f2_w, f2_w)
    branch7x7 = nnlayers.conv2d_bn(branch7x7, f3, f3_w, f3_h)
    branch7x7 = nnlayers.conv2d_bn(branch7x7, f4, f4_w, f4_h)

    branch7x7dbl = nnlayers.conv2d_bn(input, f5, f5_w, f5_w)
    branch7x7dbl = nnlayers.conv2d_bn(branch7x7dbl, f6, f6_w, f6_h)
    branch7x7dbl = nnlayers.conv2d_bn(branch7x7dbl, f7, f7_w, f7_h)
    branch7x7dbl = nnlayers.conv2d_bn(branch7x7dbl, f8, f8_w, f8_h)
    branch7x7dbl = nnlayers.conv2d_bn(branch7x7dbl, f9, f9_w, f9_h)

    branch_pool = nnlayers.avg_pooling(input, 3, strides=1, padding='same')
    branch_pool = nnlayers.conv2d_bn(branch_pool, f10, f10_w, f10_w)
    return tf.concat(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3 )




def build_model():
    x = nnlayers.conv2d_bn(x_image, 32, 3, 3, strides=2, padding='valid')
    x = nnlayers.conv2d_bn(x, 32, 3, 3, padding='valid')
    x = nnlayers.conv2d_bn(x, 64, 3, 3)
    x = nnlayers.max_pooling(x, 3, 2)
    
    x = nnlayers.conv2d_bn(x, 80, 1, 1, padding='valid')
    x = nnlayers.conv2d_bn(x, 192, 3, 3, padding='valid')
    x = nnlayers.max_pooling(x, 3, 2)
    
    x = inception_v3(x, 64, 1, 48, 1, 64, 5, 64, 1, 96, 3, 96, 3, 3, 1, 32, 1)
    x = inception_v3(x, 64, 1, 48, 1, 64, 5, 64, 1, 96, 3, 96, 3, 3, 1, 64, 1)
    x = inception_v3(x, 64, 1, 48, 1, 64, 5, 64, 1, 96, 3, 96, 3, 3, 1, 64, 1)
 
    #we've got a small inception block here.
    branch3x3 = nnlayers.conv2d_bn(x, 384, 3, 3, strides=2, padding='valid')
    branch3x3dbl = nnlayers.conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = nnlayers.conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = nnlayers.conv2d_bn(branch3x3dbl, 96, 3, 3, strides=2, padding='valid')
    branch_pool = nnlayers.max_pooling(x, 3, strides=2, padding='valid')
    x = tf.concat(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=3 )

    x = inception_v3_b(x, 192, 1, 128, 1, 128, 1, 7, 192, 7, 1, 128, 1, 128, 7, 1, 128, 1, 7, 128, 7, 1, 192, 1, 7, 192, 1)
    x = inception_v3_b(x, 192, 1, 160, 1, 160, 1, 7, 192, 7, 1, 160, 1, 160, 7, 1, 160, 1, 7, 160, 7, 1, 192, 1, 7, 192, 1)
    x = inception_v3_b(x, 192, 1, 160, 1, 160, 1, 7, 192, 7, 1, 160, 1, 160, 7, 1, 160, 1, 7, 160, 7, 1, 192, 1, 7, 192, 1)
    x = inception_v3_b(x, 192, 1, 192, 1, 192, 1, 7, 192, 7, 1, 192, 1, 192, 7, 1, 192, 1, 7, 192, 7, 1, 192, 1, 7, 192, 1)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = nnlayers.conv2d_bn(x, 192, 1, 1)
    branch3x3 = nnlayers.conv2d_bn(branch3x3, 320, 3, 3,
                          strides=2, padding='valid')

    branch7x7x3 = nnlayers.conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = nnlayers.conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = nnlayers.conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = nnlayers.conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=2, padding='valid')

    branch_pool = nnlayers.max_pooling(x, 3, strides=2, padding='valid')
    x = tf.concat(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3 )
    
    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = nnlayers.conv2d_bn(x, 320, 1, 1)

        branch3x3 = nnlayers.conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = nnlayers.conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = nnlayers.conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = tf.concat(
            [branch3x3_1, branch3x3_2],
            axis=3)

        branch3x3dbl = nnlayers.conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = nnlayers.conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = nnlayers.conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = nnlayers.conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = tf.concat(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = nnlayers.avg_pooling(
            x, 3, strides=1, padding='same')
        branch_pool = nnlayers.conv2d_bn(branch_pool, 192, 1, 1)
        x = tf.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3 )
    
    #x = layers.GlobalAveragePooling2D(name='avg_pool')(x) 
    
    # Per:https://stackoverflow.com/questions/42054451/how-do-i-do-global-average-pooling-in-tensorflow
    x = tf.reduce_mean(x, axis=[1,2])
    
    #logits = fc('loss3_classifier', pool5_drop_7x7_s1, out_nodes=100)
    logits = nnlayers.dense(x, 100, kernel_regularizer='none', activation='relu')
    return logits

k = build_model()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=k))
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)

#x_train = x_train.astype("float32")
#x_test = x_test.astype("float32")
#x_train /= 255.0
#x_test /= 255.0

sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(k, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

_x_train = _x_train.reshape(_x_train.shape[0], IMAGE_SIZE)
_x_test = _x_test.reshape(_x_test.shape[0], IMAGE_SIZE)

BATCH_SIZE = 64
TRAINING_DATASET_COUNT = 50000
# train 100 rounds on every dataset in training set.
EPOCHES = 100
BATCHES = (TRAINING_DATASET_COUNT // BATCH_SIZE) * EPOCHES

(x_small_test_batch, y_small_test_batch) = nnlayers.get_batch(_x_test, y_test, 100, 1, 10000)

with tf.Session() as tfsess:
    tf.global_variables_initializer().run()
    for i in range(BATCHES):
        (x_batch, y_batch) = nnlayers.get_batch(_x_train, y_train, BATCH_SIZE, i, 50000)
        _, loss_value = tfsess.run([train_step, cross_entropy], feed_dict={x: x_batch, y_: y_batch})
        #report loss every 10 batches
        if i % 10 == 0:
            print("batch %d, loss: %g" % (i, loss_value))

        #test after a whole epoch
        if i > 0 and i % (TRAINING_DATASET_COUNT // BATCH_SIZE) == 0:
            train_accuracy = accuracy.eval(feed_dict={x: x_small_test_batch, y_: y_small_test_batch})
            print("batch %d, loss: %g, acc: %g" % (i, loss_value, train_accuracy))

    (x_batch, y_batch) = nnlayers.get_batch(_x_train, y_train, BATCH_SIZE, i, 50000)
    train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
    print("batch final, acc: %g" % train_accuracy)