import os
from math import pi

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.layers as l
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer
from keras.regularizers import Regularizer, serialize

from keras.initializers import RandomUniform, RandomNormal, Constant
from keras.callbacks import TensorBoard, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

import numpy as np
import tensorflow as tf

import pandas as pd

import keras_metrics as km

from sklearn.datasets import make_classification

np.random.seed(27)
tf.random.set_random_seed(37)


class DistanceRegularizer(Regularizer):
    def __init__(self, l):
        self.l = K.cast_to_floatx(l)

    def __call__(self, x):
        number_of_features = x.get_shape()[0].value
        output_length = x.get_shape()[1].value
        summary = K.cast_to_floatx(0)
        for feature in range(number_of_features):
            for i in range(output_length):
                cmin = 0
                for j in range(output_length):
                    if K.square(x[feature][i] - x[feature][j]) < cmin:
                        cmin = K.square(x[feature][i] - x[feature][j])
                summary += cmin
        summary /= K.cast_to_floatx(output_length)
        return -1 * self.l * summary

    def get_config(self):
        config = {'l': self.l}
        base_config = super(DistanceRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProbabilityRegularizer(Regularizer):
    def __init__(self, p=0.):
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        norm_x = K.sum(K.abs(x), axis=2)
        diff_norm = K.ones(norm_x.shape[1:])
        diff_norm = K.square(norm_x - diff_norm)
        return self.p * K.mean(K.mean(diff_norm, axis=1))

    def get_config(self):
        return {'p': float(self.p)}


class ProbabilityRegularization(Layer):
    def __init__(self, p=0., **kwargs):
        super(ProbabilityRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.p = p
        self.activity_regularizer = ProbabilityRegularizer(self.p)

    def get_config(self):
        config = {'p': self.p}
        base_config = super(ProbabilityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class EntropyRegularizer(Regularizer):
    def __init__(self, p=0.):
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        log_data = K.log(1e-10 + K.abs(x))
        entropy = K.abs(K.sum(log_data * K.abs(x), axis=2))
        return self.p * K.mean(entropy)

    def get_config(self):
        return {'p': float(self.p)}


class EntropyRegularization(Layer):
    def __init__(self, p=0., **kwargs):
        super(EntropyRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.p = p
        self.activity_regularizer = EntropyRegularizer(self.p)

    def get_config(self):
        config = {'p': self.p}
        base_config = super(EntropyRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class DiscretizationLayer(Layer):
    def __init__(self, output_dim, init_means, **kwargs):
        self.output_dim = output_dim
        self.init_means = init_means
        super(DiscretizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=Constant(self.init_means),#'random_uniform',
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.coefs = self.add_weight(name='coefs',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=RandomUniform(minval=0, maxval=1),
                                      trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        source_matrix = K.repeat(inputs, self.output_dim)
        source_matrix = K.permute_dimensions(source_matrix, (0, 2, 1))
        source_matrix = K.square(source_matrix - self.kernel)

        sigma_square = 1e-4 + K.square(self.sigmas)

        source_matrix = K.exp(-1 * sigma_square * source_matrix)

        normalization = 1 / K.sqrt(2 * pi * sigma_square)
        source_matrix *= normalization

        #print inputs.shape
        summary = K.expand_dims(K.sum(1e-10 + K.abs(self.coefs), axis=1), axis=1)
        normalization_v_prob = K.repeat(summary, self.output_dim)
        normalization_v_prob = K.squeeze(normalization_v_prob, axis=2)

        source_matrix = source_matrix * (K.abs(self.coefs)) / normalization_v_prob

        return source_matrix

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim,)

    def get_config(self):
        config = {'init_means': self.init_means,}
        base_config = super(DiscretizationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def softmax(x):
    x_p = np.sin(x) + np.cos(1e-4 + x * x)
    x_p = [x_p[i] + x_p[i + 1] for i in range(0, len(x_p), 2)]
    target = np.zeros(len(x_p))
    target[np.argmax(x_p)] = 1
    return target

input_shape = 30
layer_steps = 20

means = np.linspace(-10, 10, input_shape) + np.random.randint(-1, 1, input_shape)
covs = np.eye(input_shape)

size = 80000
classes = 2

data = pd.read_csv('./input/creditcard.csv')
#columns = list(data.columns.values)
#print columns
#print data.shape
#exit(0)
#for cname in columns:
#    data[cname].fillna(data[cname].mean(), inplace=True)

X = data.drop(columns=['Class']).values
y = data['Class'].values
#y = pd.read_csv('./input/ytrain.csv')
#y = y['x'].values

#y_new = []
#for item in y:
#    d = np.zeros(classes)
#    d[item] = 1
#    y_new.append(d)

#y = np.array(y_new)
print y.shape, X.shape

linspace_init = False
rng_init = True

bn = False
bn_nm = True
nm = True
apply_regularization = False

if not linspace_init:
    percentiles = np.array([np.random.uniform(min(X[:, i]), max(X[:, i]), size=layer_steps) for i in range(input_shape)])
else:
    if rng_init:
        percentiles = np.random.uniform(-1, 1, (input_shape, layer_steps))
    else:
        percentiles = np.array([np.linspace(min(X[:, i]), max(X[:, i]), layer_steps) for i in range(input_shape)])

#output_dim = y.shape[1]

dropout_rate = 0.1

fname = 'test' + str(layer_steps) + '_'
if nm:
    fname += 'disc_'
    if apply_regularization:
        fname += 'reg_'
    else:
        fname += 'no_reg_'
    if linspace_init:
        fname += 'ls_init'
    else:
        if not rng_init:
            fname += 'rng_mm'
        else:
            fname += 'rng_uf'
else:
    fname += 'dense'

print fname

callbacks = [TensorBoard('./' + fname + '/')]

callbacks.append(ReduceLROnPlateau(verbose=1, patience=5, min_lr=1e-6, factor=0.5))

neurons = layer_steps * input_shape

input = l.Input(shape=(input_shape,))
next = input
if nm:
    if bn_nm:
        next = l.BatchNormalization()(next)
    next = DiscretizationLayer(layer_steps, percentiles)(next)
    if apply_regularization:
        next = EntropyRegularization(1.)(next)
        #next = ProbabilityRegularization(1.)(next)
    next = l.Flatten()(next)
else:
    next = l.Dense(neurons, activation='elu')(next)
next = l.Dense(100, activation='elu')(next)
if bn:
    next = l.BatchNormalization()(next)
next = l.Dropout(dropout_rate)(next)
next = l.Dense(50, activation='elu')(next)
if bn:
    next = l.BatchNormalization()(next)
next = l.Dropout(dropout_rate)(next)
next = l.Dense(1, activation='sigmoid')(next)

model = Model(input=input, output=next)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', km.f1_score()])

#print model.predict(X)

model.summary()

print float(sum(y)) / len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=29)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

print class_weights

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250, batch_size=4096,
          callbacks=callbacks, class_weight=class_weights)

