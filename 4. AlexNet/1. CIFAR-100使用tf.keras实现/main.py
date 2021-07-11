import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

BATCH_SIZE = 64
EPOCH = 20

# label_mode: one of "fine", "coarse". If it is "fine" the category labels are the fine-grained labels,
# if it is "coarse" the output labels are the coarse-grained superclasses. we've used fine here, so it is divided
# into 100 categories.
# https://keras.io/api/datasets/cifar100/
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

# one hot
y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

model = Sequential()
# CIFAR-100: This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 100
# fine-grained classes that are grouped into 20 coarse-grained classes. original input 64x64x3 --> conv 1 : 11x11,
# so input=32x32x3, strides=4, 96 kernels
model.add(Conv2D(96, (5, 5), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu',
                 kernel_initializer='uniform'))
#max pooling layer: 3x3 strides = 2
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#conv2: 5x5, pad=2, 256 kernels
model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
#max pooling layer: 3x3 strides = 2
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#conv3: 3x3 pad = 1, 384 kernels
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform' ))
#conv4: same as conv3
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform' ))
#conv5: 3x3 pad=1, 256 kernels
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform' ))
#max pooling layer: 3x3, strides = 2
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#flatten to final fc layer
model.add(Flatten())
#full connecting layer 1: 4096
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
#fc2: 4096
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
#softmax: 1000
model.add(Dense(100, activation='softmax'))

opt = keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, decay=0.0005, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0

#model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test), shuffle=True)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)

# using processed image data to train the model
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCH,
                    validation_data=(x_test, y_test),
                    workers=4)

scores = model.evaluate(x_test, y_test, verbose=1)
print('loss ', scores[0], " accuracy ", scores[1])
