import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np

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

def conv_2d(input, kernel_size, filter_count, strides = 1):
    return tf.compat.v1.layers.conv2d(input, filters=filter_count, kernel_size=[kernel_size, kernel_size], use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01), strides=[strides, strides],
                                     bias_initializer=tf.constant_initializer(0.01), activation=tf.nn.relu, padding='SAME')

def max_pooling(input, kernel_size, strides):
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1], padding="VALID")

def dense(input, units, stddev=0.04,  activation=tf.nn.relu):
    return tf.layers.dense(inputs=input, units=units, use_bias=True,
                            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                            bias_initializer=tf.constant_initializer(0.1), activation=activation)
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
LABEL_BYTES = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH  # 3072

x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])
x_image = tf.reshape(x, [-1, 32, 32, 3])

# CIFAR-100:
# input=32x32x3, ksize=5, 96 kernels
h_conv1 = conv_2d(x_image, 5, 96)
print("conv1," , np.shape(h_conv1))
#max pooling layer: 3x3 strides = 2
h_pool1 = max_pooling(h_conv1, 3, 2)
print("pool1," , np.shape(h_pool1))
#conv2: 5x5, strides=2, 256 kernels
h_conv2 = conv_2d(h_pool1, 5, 256, strides=2)
print("conv2," , np.shape(h_conv2))
#max pooling layer: 3x3 strides = 2
h_pool2 = max_pooling(h_conv2, 3, 2)
print("pool2," ,np.shape(h_pool2))
#conv3: 3x3 strides = 1, 384 kernels
h_conv3 = conv_2d(h_pool2, 3, 384)
print("conv3," , np.shape(h_conv3))
#conv4: same as conv3
h_conv4 = conv_2d(h_conv3, 3, 384)
print("conv4," , np.shape(h_conv4))
#conv5: 3x3 pad=1, 256 kernels
h_conv5 = conv_2d(h_conv3, 3, 256)
print("conv5," , np.shape(h_conv5))
#max pooling layer: 3x3, strides = 2
h_pool3 = max_pooling(h_conv5, 3, 2)
print("pool3," , np.shape(h_pool3))
#flatten to final fc layer
h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
print("flatten," , np.shape(h_pool3_flat))
#full connecting layer 1: 4096
h_fc1 = dense(h_pool3_flat, 4096, 0.04)
print("fc1," , np.shape(h_fc1))
#0.50 dropout
h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-0.15)
#fc2: 4096
h_fc2 = dense(h_fc1_drop, 4096, 0.02)
print("fc2," , np.shape(h_fc2))
#0.50 dropout
h_fc2_drop = tf.nn.dropout(h_fc2, rate=1-0.15)
#softmax: 1000
logits = dense(h_fc2_drop, 100, 0.001, activation=tf.nn.softmax)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)

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

(x_small_test_batch, y_small_test_batch) = get_batch(x_test, y_test, 500, 1, 10000)

BATCH_SIZE = 200
TRAINING_DATASET_COUNT = 50000
#train 100 rounds on every dataset in training set.
EPOCHES = (TRAINING_DATASET_COUNT // BATCH_SIZE) * 100

for i in range(EPOCHES):
    (x_batch, y_batch) = get_batch(x_train, y_train, BATCH_SIZE, i, 50000)
    if i > 0 and i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: x_small_test_batch, y_: y_small_test_batch})
        print("step %d, acc: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y_: y_batch})

train_accuracy = accuracy.eval(feed_dict={x: x_train, y_: y_test})
print("step final, acc: %g" % train_accuracy)
