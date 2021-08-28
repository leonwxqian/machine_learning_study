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


IMAGE_WIDTH = 224
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH

_x_train = []
_x_test = []
for x_tr in x_train:
    _x_train.append(cv2.resize(x_tr, (IMAGE_WIDTH, IMAGE_WIDTH)))
for x_te in x_test:
    _x_test.append(cv2.resize(x_te, (IMAGE_WIDTH, IMAGE_WIDTH)))
print(np.array(_x_train).shape)
_x_train = np.array(_x_train).reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, 3)
_x_test = np.array(_x_test).reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, 3)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_WIDTH, 3])
 


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    x = nnlayers.conv2d_bn(input_tensor, filters1, 1, 1, kernel_initializer='he_normal', activation='relu', padding='valid') #is padding correct??
    x = nnlayers.conv2d_bn(x, filters2, 1, 1, kernel_initializer='he_normal', activation='relu', padding='same')
    x = nnlayers.conv2d_bn(x, filters3, 1, 1, kernel_initializer='he_normal', activation='relu', padding='valid')
    #x = x + input_tensor
    x = tf.keras.layers.add([x, input_tensor])
    x = tf.nn.relu(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3 
    x = nnlayers.conv2d_bn(input_tensor, filters1, 1, 1, strides=strides[0], kernel_initializer='he_normal', activation='relu', padding='valid') #is padding correct??
    x = nnlayers.conv2d_bn(x, filters2, 1, 1, kernel_initializer='he_normal', activation='relu', padding='same')
    x = nnlayers.conv2d_bn(x, filters3, 1, 1, kernel_initializer='he_normal', kernel_regularizer='none', padding='valid')
    shortcut = nnlayers.conv2d_bn(input_tensor, filters3, 1, 1, strides=strides[0], kernel_initializer='he_normal', kernel_regularizer='none', padding='valid')
    x = tf.keras.layers.add([x, shortcut])

    x = tf.nn.relu(x)
    return x


def build_model(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    bn_axis = 3
    

    x = nnlayers.zero_padding(x_image, padding=(3, 3), )
    
    x = nnlayers.conv2d_bn(x, 64, 7, 7, strides=2, kernel_initializer='he_normal', activation='relu', padding='valid') #is padding correct??
 
    x = nnlayers.zero_padding(x, padding=(1, 1))
    
    x = nnlayers.max_pooling(x, 3, strides=2, padding='valid') 
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Per:https://stackoverflow.com/questions/42054451/how-do-i-do-global-average-pooling-in-tensorflow
    x = tf.reduce_mean(x, axis=[1,2])
    x = nnlayers.dense(x, 100, kernel_regularizer='none', activation='relu') #no softmax here!
    return x

k = build_model()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=k))
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)


sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(k, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

_x_train = _x_train.reshape(_x_train.shape[0], IMAGE_SIZE)
_x_test = _x_test.reshape(_x_test.shape[0], IMAGE_SIZE)

_x_train = _x_train.astype("float32")
_x_test = _x_test.astype("float32")
_x_train /= 255.0
_x_test /= 255.0

BATCH_SIZE = 32
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