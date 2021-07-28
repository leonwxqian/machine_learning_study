import tensorflow.compat.v1 as tf
import numpy as np
 
tf.disable_v2_behavior()


def conv_2d(input, kernel_size, filter_count, strides=1, kernel_regularizer='none'):
    if kernel_regularizer == 'none':
        return tf.compat.v1.layers.conv2d(input, filters=filter_count, kernel_size=[kernel_size, kernel_size],
                                          use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          strides=[strides, strides],
                                          bias_initializer=tf.constant_initializer(0.01), activation=tf.nn.relu,
                                          padding='SAME')
    elif kernel_regularizer == 'l2':
        return tf.compat.v1.layers.conv2d(input, filters=filter_count, kernel_size=[kernel_size, kernel_size],
                                          use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          strides=[strides, strides],
                                          bias_initializer=tf.constant_initializer(0.01), activation=tf.nn.relu,
                                          padding='SAME',
                                          kernel_regularizer=tf.keras.regularizers.L2(0.0005))

										  
def conv_2dnn(input, kernel_size, filter_count, strides=1, kernel_regularizer='none'):
    print(input.shape[-1])
    w_conv = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input.shape[-1], filter_count], stddev=0.01))
    b_conv = tf.Variable(tf.constant(0.1, shape=[filter_count]))

    if kernel_regularizer == 'none':
        return tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv)
    elif kernel_regularizer == 'l2':
        return tf.nn.l2_normalize(tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv), dim=0, epsilon=0.0005)


def max_pooling(input, kernel_size, strides=0, padding="VALID"):
    if strides == 0:
        strides = kernel_size
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1],
                          padding=padding)

#use tf.nn.xw_plus_b
def dense(input, units, stddev=0.04, bias=0.01, kernel_regularizer='none', activation='relu'):
    w_fc = tf.Variable(
        tf.truncated_normal([input.shape[-1], units], stddev=stddev))  # kernel initializer=truncated_normal
    b_fc = tf.Variable(tf.constant(bias, shape=[units]))  # bias_initializer=constant 0.01
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


def flatten(input):
    print(np.shape(input))
    return tf.reshape(input, [-1, input.shape[1] * input.shape[2] * input.shape[3]])

