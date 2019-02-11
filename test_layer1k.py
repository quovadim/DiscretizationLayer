import os
from math import pi
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import keras.layers as l
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer
from keras.regularizers import Regularizer, serialize

from keras.initializers import RandomUniform, RandomNormal, Constant, Ones
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.constraints import NonNeg, MinMaxNorm

import numpy as np
import tensorflow as tf

import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from keras_metrics import recall, precision

np.random.seed(27)
tf.random.set_random_seed(37)


class DiscretizationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=RandomUniform(minval=-2, maxval=2),
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=RandomNormal(mean=1, stddev=0.1),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.mix = self.add_weight(name='mix',
                                   shape=(input_shape[1], self.output_dim,),
                                   initializer=RandomNormal(1, 0.1),
                                   constraint=NonNeg(),
                                   trainable=True)

        self.temperature = self.add_weight(name='temperature',
                                           shape=(input_shape[1], 1,),
                                           initializer=RandomNormal(1, 0.1),
                                           trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        #dimension_in = inputs.shape[1]
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        means = self.kernel#tf.tile(self.kernel, [dimension_in, 1])
        bias = self.mix#tf.tile(self.mix, [dimension_in, 1])
        sigma = self.sigmas#tf.tile(self.sigmas, [dimension_in, 1])
        temp = self.temperature#tf.tile(self.temperature, [dimension_in, 1])
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


def make_net(neurons, input_shape, disc):
    input = l.Input(shape=(input_shape,))
    next = input
    if disc:
        next = DiscretizationLayer(input_shape)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    if disc:
        next = DiscretizationLayer(neurons)(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    if disc:
        next = DiscretizationLayer(neurons / 2)(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    if disc:
        next = DiscretizationLayer(neurons / 4)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(1, activation='sigmoid')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=5e-3), loss='binary_crossentropy', metrics=['accuracy', recall(), precision()])

    return model

higgs_dataset = {'file': './HIGGS.csv', 'input': 28, 'y label': '0'}
susy_dataset = {'file': './SUSY4.csv', 'input': 18, 'y label': '0'}
htru_dataset = {'file': './HTRU_2.csv', 'input': 8, 'y label': '8'}

cdataset = susy_dataset

input_shape = cdataset['input']
layer_steps = input_shape * 3

classes = 2

print 'Reading...'
X = pd.read_csv(cdataset['file'], nrows=500000)
print X.head()

y = X[cdataset['y label']].values
X = X.drop(columns=[cdataset['y label']]).values

dropout_rate = 0.0

i = input_shape
k = 30
z = layer_steps

neurons = 256

model = make_net(neurons, input_shape, True)
model_dense = make_net(neurons * 3, input_shape, False)

model.summary()
model_dense.summary()

epoch_steps = 20
callbacks_disc = [ReduceLROnPlateau(verbose=1, patience=10, min_lr=5e-5, factor=0.5),
                  TensorBoard('./new/')]
callbacks_dense = [ReduceLROnPlateau(verbose=1, patience=10, min_lr=5e-5, factor=0.5), TensorBoard('./dense/')]
'''
python /home/vadim/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir LS:./Laplace_softmax/,GS:./Grumbel_softmax/,NS:./Normal_softmax/,LL:./Laplace_l1/,GL:./Grumbel_l1/,NL:./Normal_l1/,LN:./Laplace_none/,GN:./Grumbel_none/,NN:./Normal_none/,dense:./old/
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27, train_size=0.85)
for i in range(500 / epoch_steps):
    print '---------------------------------------------------------------------------------------------'
    model_dense.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=(i + 1) * epoch_steps,
                    initial_epoch=i * epoch_steps, batch_size=1024, callbacks=callbacks_dense, verbose=1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=(i + 1) * epoch_steps,
              initial_epoch=i * epoch_steps, batch_size=1024, callbacks=callbacks_disc, verbose=1)
