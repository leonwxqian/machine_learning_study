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
import cv2


tf.disable_v2_behavior()

# label_mode: one of "fine", "coarse". If it is "fine" the category labels are the fine-grained labels,
# if it is "coarse" the output labels are the coarse-grained superclasses. we've used fine here, so it is divided
# into 100 categories.
# https://keras.io/api/datasets/cifar100/
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")


#the original data for inception is 224x224x3. but cifar has 32x32x3 which is TOO SMALL for it..
_x_train = []
_x_test = []
for x_tr in x_train:
    _x_train.append(cv2.resize(x_tr, (224, 224)))
for x_te in x_test:
    _x_test.append(cv2.resize(x_te, (224, 224)))
print(np.array(_x_train).shape)
_x_train = np.array(_x_train).reshape(-1, 224, 224, 3)
_x_test = np.array(_x_test).reshape(-1, 224, 224, 3)


#0.0013
weight_decay = 0.01
def inception_model(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay))(input)

    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(weight_decay))(input)

    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay))(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(weight_decay))(input)

    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(weight_decay))(conv_5x5_reduce)

    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)

    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same',
                          activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(maxpool)

    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend

    return inception_output



input = Input(shape=(224, 224, 3))
#strides=2 will make the image very small. leave them as 1.
conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input)

maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)

conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(maxpool1_3x3_s2)

#, strides=(2, 2) is added.
conv2_3x3 = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(conv2_3x3_reduce)
                  # )(conv2_3x3_reduce)


maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)

#the network is too large for CIFAR-100. delete 2 inception models will lower the chance to overfitting.
inception_3a = inception_model(input=maxpool2_3x3_s2, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                               filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

inception_3b = inception_model(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                               filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_3b)

inception_4a = inception_model(input=maxpool3_3x3_s2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                               filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)

inception_4b = inception_model(input=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                               filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

inception_4c = inception_model(input=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                               filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

inception_4d = inception_model(input=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                               filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)

inception_4e = inception_model(input=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                               filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)

maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_4e)

inception_5a = inception_model(input=maxpool4_3x3_s2, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                               filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)

inception_5b = inception_model(input=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
                               filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)

#don't change 7x7 to 2x2, 7x7 will keep the output as original if it is smaller than 7x7.
#but if we set it to 2x2, it will /2 almost everytime, it lowers the output, and the output will be as small
#as 2x2 or even 1x1, bad for us.
averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(inception_5b)

drop1 = Dropout(rate=0.4)(averagepool1_7x7_s1)
flatten = Flatten()(drop1)
#flatten = Flatten()(averagepool1_7x7_s1)

linear = Dense(units=100, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(flatten)
last = linear

model = Model(inputs=input, outputs=last)
model.summary()
BATCH_SIZE = 64
EPOCH = 100

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(_x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(_x_test, y_test), shuffle=True)

#no good here..
#datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
#datagen.fit(_x_train)

# using processed image data to train the model
#model.fit_generator(datagen.flow(_x_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCH, validation_data=(_x_test, y_test))