'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer, BatchNormalization
from keras.regularizers import Regularizer, serialize

from keras.initializers import RandomUniform, RandomNormal, Constant, Ones
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.constraints import NonNeg, MinMaxNorm

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow as tf


class DiscretizationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.output_dim),
                                      initializer=RandomUniform(minval=-2, maxval=2),
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(1, self.output_dim),
                                      initializer=RandomNormal(mean=1, stddev=0.1),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.mix = self.add_weight(name='mix',
                                   shape=(1, self.output_dim,),
                                   initializer=RandomNormal(1, 0.1),
                                   constraint=NonNeg(),
                                   trainable=True)

        self.temperature = self.add_weight(name='temperature',
                                           shape=(1, 1,),
                                           initializer=RandomNormal(1, 0.1),
                                           trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        dimension_in = 288
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        print(x.shape, inputs.shape)
        #print(self.kernel)
        means = tf.tile(self.kernel, [dimension_in, 1])#self.kernel#
        bias = tf.tile(self.mix, [dimension_in, 1])#self.mix#
        sigma = tf.tile(self.sigmas, [dimension_in, 1])#self.sigmas#
        temp = tf.tile(self.temperature, [dimension_in, 1])#self.temperature#
        #x = tf.exp(-1 * tf.abs(x - means) / (1e-5 + sigma)) / (1e-5 + 2 * sigma)
        #x_ml = x / tf.reduce_mean(x, axis=2, keep_dims=True)
        x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
        x = x / (1e-7 + tf.abs(temp))
        x_ml = tf.nn.softmax(x)
        x = x_ml * x_expanded
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)

batch_size = 512
num_classes = 10
epochs = 150

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
if True:
    model.add(BatchNormalization())
    model.add(DiscretizationLayer(256))
else:
    model.add(BatchNormalization())
    model.add(Dense(256))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(rotation_range=45,
                             width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, vertical_flip=True)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=datagen.flow(x_test, y_test, batch_size=batch_size),
                    validation_steps=len(x_test) // batch_size,
                    callbacks=[TensorBoard('./old/')])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])