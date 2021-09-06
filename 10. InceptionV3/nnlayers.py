import tensorflow.compat.v1 as tf
import numpy as np
 
tf.disable_v2_behavior()

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


def zero_padding(input, padding=3, data_format=None):
    return tf.keras.layers.ZeroPadding2D(padding=(padding, padding), data_format=None)(input)


def conv2d_bn(input, filter_count, kernel_row, kernel_col, padding='SAME', strides=1, kernel_regularizer='none', l2_value=0.0005, activation='none', kernel_initializer='glorot_normal', axis=3):
    conv = complex_conv_2d(input, filter_count, kernel_row, kernel_col, strides=strides, padding=padding,kernel_regularizer=kernel_regularizer, l2_value=l2_value, activation=activation, kernel_initializer=kernel_initializer)
    bn = batch_normalization(conv, axis)
    return tf.nn.relu(bn)

def complex_conv_2d(input, filter_count, kernel_row, kernel_col, strides=1, padding='SAME',kernel_initializer='glorot_normal', kernel_regularizer='none', l2_value=0.0005, activation='relu'):
    kern = None
    if kernel_initializer == 'none':
        kern = None
    elif kernel_initializer == 'glorot_normal':
        kern = tf.compat.v1.keras.initializers.glorot_normal()
    elif kernel_initializer == 'he_normal':
        kern = tf.compat.v1.keras.initializers.he_normal()

    kr = None
    if kernel_regularizer == 'none':
        kr = None
    elif kernel_regularizer == 'l2':
        kr = tf.keras.regularizers.L2(l2_value)

    activfn = None
    if activation == 'none':
        activfn = None
    elif activation == 'relu':
        activfn = tf.nn.relu
 
    #print(filter_count, kernel_row, kernel_col, kern, strides, activfn, padding, kr)
    return tf.compat.v1.layers.conv2d(input, filters=filter_count, kernel_size=[kernel_row, kernel_col],
                                          use_bias=True,
                                          kernel_initializer=kern,
                                          strides=[strides, strides],
                                          bias_initializer=tf.constant_initializer(0.0), activation=activfn,
                                          padding=padding.upper(),
                                          kernel_regularizer=kr)


def conv_2d(input, kernel_size, filter_count, strides=1, padding='SAME',kernel_initializer = 'glorot_normal', kernel_regularizer='none', l2_value=0.0005, activation='relu'):
    if kernel_initializer == 'none':
        kern = None
    elif kernel_initializer == 'glorot_normal':
        kern = tf.compat.v1.keras.initializers.glorot_normal()
    elif kernel_initializer == 'he_normal':
        kern = tf.compat.v1.keras.initializers.he_normal()
    
    if kernel_regularizer == 'none':
        kr = None
    elif kernel_regularizer == 'l2':
        kr = tf.keras.regularizers.L2(l2_value)
        
    if activation == 'none':
        activfn = None
    elif activation == 'relu':
        activfn = tf.nn.relu
 
 
    return tf.compat.v1.layers.conv2d(input, filters=filter_count, kernel_size=[kernel_size, kernel_size],
                                          use_bias=True,
                                          kernel_initializer=kern,
                                          strides=[strides, strides],
                                          bias_initializer=tf.constant_initializer(0.0), activation=activfn,
                                          padding=padding.upper(),
                                          kernel_regularizer=kr)

										  
def conv_2dnn(input, kernel_size, filter_count, strides=1, padding='SAME', kernel_regularizer='none'):
    print(input.shape[-1])
    w_conv = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input.shape[-1], filter_count], stddev=0.01))
    b_conv = tf.Variable(tf.constant(0.1, shape=[filter_count]))

    if kernel_regularizer == 'none':
        return tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding=padding.upper()) + b_conv)
    elif kernel_regularizer == 'l2':
        return tf.nn.l2_normalize(tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, strides, strides, 1], padding=padding.upper()) + b_conv), dim=0, epsilon=0.0005)


def max_pooling(input, kernel_size, strides=0, padding="SAME"): #if anything error happends, check if padding should be VALID.
    if strides == 0:
        strides = kernel_size
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1],
                          padding=padding.upper())

def avg_pooling(input, kernel_size, strides=0, padding="SAME"):
    if strides == 0:
        strides = kernel_size
    return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1],
                          padding=padding.upper())

def lrn(inputs, depth_radius=5, alpha=0.0001, beta=0.75):
    return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                  alpha=alpha, beta=beta)


#use tf.nn.xw_plus_b
def dense(input, units, stddev=0.04, bias=0.0, kernel_regularizer='none', activation='relu', epsilon=0.0005):
    shape = input.get_shape()
    if len(shape) == 4:  # x is 4D tensor
        size = shape[1].value * shape[2].value * shape[3].value
    else:  # x has already flattened
        size = shape[-1].value
    #w_fc = tf.Variable(
    #    tf.truncated_normal([input.shape[-1], units], stddev=stddev))  # kernel initializer=truncated_normal
    w_fc = tf.get_variable('weights',
                            shape=[size, units],
                            initializer=tf.compat.v1.keras.initializers.glorot_normal())
   # b_fc = tf.Variable(tf.constant(bias, shape=[units]))  # bias_initializer=constant 0.01
    #dont give any value to bias in dense.
    b_fc = tf.get_variable('biases',
                            shape=[units],
                            initializer=tf.constant_initializer(bias))
    #logits = tf.nn.xw_plus_b(x=input, weights=w_fc, biases=b_fc)
    flat_x = tf.reshape(input, [-1, size])
    logits = tf.nn.bias_add(tf.matmul(flat_x, w_fc), b_fc)

    if kernel_regularizer == 'none' and activation == 'none':
        return logits
    elif kernel_regularizer == 'none' and activation == 'relu':
        return tf.nn.relu(logits)
    elif kernel_regularizer == 'l2' and activation == 'relu':
        return tf.nn.l2_normalize(tf.nn.relu(logits), dim=0, epsilon=epsilon)
    elif kernel_regularizer == 'none' and activation == 'softmax':
        return tf.nn.softmax(logits)


def batch_normalization(inputs, axes=0):
    mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.get_shape())-1)))
    return tf.nn.batch_normalization(
        inputs, mean, variance, None, None, 1e-12, name=None  # try 0.001 also
    )


def dropout(input, percentage):
    return tf.nn.dropout(input, rate=1 - percentage)


def flatten(input):
    print(np.shape(input))
    return tf.reshape(input, [-1, input.shape[1] * input.shape[2] * input.shape[3]])

