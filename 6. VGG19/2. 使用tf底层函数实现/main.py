import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
import nnlayers

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


IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH  # 3072

x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = nnlayers.conv_2d(x_image, 3, 64)
h_norm1 = nnlayers.batch_normalization(h_conv1)
h_dropout1 = nnlayers.dropout(h_norm1, 0.3)
print("layer1, ", np.shape(h_dropout1))

h_conv2 = nnlayers.conv_2d(h_dropout1, 3, 64, kernel_regularizer='l2')
h_norm2 = nnlayers.batch_normalization(h_conv2)
# according to https://keras.io/zh/layers/pooling/, strides is equal to pool_size by default.
h_pool2 = nnlayers.max_pooling(h_norm2, 2, 2)
print("layer2, ", np.shape(h_pool2))

h_conv3 = nnlayers.conv_2d(h_pool2, 3, 128, kernel_regularizer='l2')
h_norm3 = nnlayers.batch_normalization(h_conv3)
h_dropout3 = nnlayers.dropout(h_norm3, 0.4)
print("layer3, ", np.shape(h_dropout3))

# should be layer#4... i made a silly mistake by not changing its index when writing this..
h_conv3_b = nnlayers.conv_2d(h_dropout3, 3, 128, kernel_regularizer='l2')
h_norm3_b = nnlayers.batch_normalization(h_conv3_b)
h_pool3_b = nnlayers.max_pooling(h_norm3_b, 2, 2)
print("layer4, ", np.shape(h_pool3_b))

h_conv4 = nnlayers.conv_2d(h_pool3_b, 3, 256, kernel_regularizer='l2')
h_norm4 = nnlayers.batch_normalization(h_conv4)
h_dropout4 = nnlayers.dropout(h_norm4, 0.4)

h_conv5 = nnlayers.conv_2d(h_dropout4, 3, 256, kernel_regularizer='l2')
h_norm5 = nnlayers.batch_normalization(h_conv5)
h_dropout5 = nnlayers.dropout(h_norm5, 0.4)

h_conv5_vgg19 = nnlayers.conv_2d(h_dropout5, 3, 256, kernel_regularizer='l2')
h_norm5_vgg19 = nnlayers.batch_normalization(h_conv5_vgg19)
h_dropout5_vgg19 = nnlayers.dropout(h_norm5_vgg19, 0.4)

h_conv6 = nnlayers.conv_2d(h_dropout5_vgg19, 3, 256, kernel_regularizer='l2')
h_norm6 = nnlayers.batch_normalization(h_conv6)
h_pool6 = nnlayers.max_pooling(h_norm6, 2, 2)

h_conv7 = nnlayers.conv_2d(h_pool6, 3, 512, kernel_regularizer='l2')
h_norm7 = nnlayers.batch_normalization(h_conv7)
h_dropout7 = nnlayers.dropout(h_norm7, 0.4)

h_conv8 = nnlayers.conv_2d(h_dropout7, 3, 512, kernel_regularizer='l2')
h_norm8 = nnlayers.batch_normalization(h_conv8)
h_dropout8 = nnlayers.dropout(h_norm8, 0.4)

h_conv8_vgg19 = nnlayers.conv_2d(h_dropout8, 3, 512, kernel_regularizer='l2')
h_norm8_vgg19 = nnlayers.batch_normalization(h_conv8_vgg19)
h_dropout8_vgg19 = nnlayers.dropout(h_norm8_vgg19, 0.4)

h_conv9 = nnlayers.conv_2d(h_dropout8_vgg19, 3, 512, kernel_regularizer='l2')
h_norm9 = nnlayers.batch_normalization(h_conv9)
h_pool9 = nnlayers.max_pooling(h_norm9, 2, 2)

h_conv10 = nnlayers.conv_2d(h_pool9, 3, 512, kernel_regularizer='l2')
h_norm10 = nnlayers.batch_normalization(h_conv10)
h_dropout10 = nnlayers.dropout(h_norm10, 0.4)

h_conv11 = nnlayers.conv_2d(h_dropout10, 3, 512, kernel_regularizer='l2')
h_norm11 = nnlayers.batch_normalization(h_conv11)
h_dropout11 = nnlayers.dropout(h_norm11, 0.4)

h_conv11_vgg19 = nnlayers.conv_2d(h_dropout11, 3, 512, kernel_regularizer='l2')
h_norm11_vgg19 = nnlayers.batch_normalization(h_conv11_vgg19)
h_dropout11_vgg19 = nnlayers.dropout(h_norm11_vgg19, 0.4)

h_conv12 = nnlayers.conv_2d(h_dropout11_vgg19, 3, 512, kernel_regularizer='l2')
h_norm12 = nnlayers.batch_normalization(h_conv12)
h_pool12 = nnlayers.max_pooling(h_norm12, 2, 2)
h_dropout12 = nnlayers.dropout(h_pool12, 0.5)
print(np.shape(h_dropout12))
h_flatten = nnlayers.flatten(h_dropout12, 512)

h_dense1 = nnlayers.dense(h_flatten, 512, kernel_regularizer='l2')
h_norm_d1 = nnlayers.batch_normalization(h_dense1)
h_dropout_d1 = nnlayers.dropout(h_norm_d1, 0.5)

# h_dense2 = nnlayers.dense(h_dropout_d1, 100, kernel_regularizer='none', activation='softmax')
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