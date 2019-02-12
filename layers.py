from keras.layers import Layer
from keras.initializers import RandomUniform, RandomNormal
from keras.constraints import NonNeg

import numpy as np
import tensorflow as tf

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
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        means = self.kernel
        bias = self.mix
        sigma = self.sigmas
        temp = self.temperature
        x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
        x = x / (1e-7 + tf.abs(temp))
        x_ml = tf.nn.softmax(x)
        x = x_ml * x_expanded
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)


class DiscretizationLayerLite(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayerLite, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayerLite, self).build(input_shape)
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
        dimension_in = inputs.shape[1]
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        means = tf.tile(self.kernel, [dimension_in, 1])
        bias = tf.tile(self.mix, [dimension_in, 1])
        sigma = tf.tile(self.sigmas, [dimension_in, 1])
        temp = tf.tile(self.temperature, [dimension_in, 1])
        x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
        x = x / (1e-7 + tf.abs(temp))
        x_ml = tf.nn.softmax(x)
        x = x_ml * x_expanded
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)


class DiscretizationLayerLaplace(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayerLaplace, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayerLaplace, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=RandomUniform(minval=-2, maxval=2),
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=RandomNormal(mean=1, stddev=0.1),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        means = self.kernel
        sigma = self.sigmas
        x = tf.exp(-1 * tf.abs(x - means) / (1e-5 + sigma)) / (1e-5 + 2 * sigma)
        x_ml = x / tf.reduce_mean(x, axis=2, keep_dims=True)
        x = x_ml * x_expanded
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)


class DiscretizationLayerLaplaceLite(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayerLaplaceLite, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiscretizationLayerLaplaceLite, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.output_dim),
                                      initializer=RandomUniform(minval=-2, maxval=2),
                                      trainable=True)

        self.sigmas = self.add_weight(name='sigma',
                                      shape=(1, self.output_dim),
                                      initializer=RandomNormal(mean=1, stddev=0.1),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        dimension_in = inputs.shape[1]
        dimension_out = self.output_dim
        x_expanded = tf.expand_dims(inputs, -1)
        x = tf.tile(x_expanded, [1, 1, dimension_out])
        means = tf.tile(self.kernel, [dimension_in, 1])
        sigma = tf.tile(self.sigmas, [dimension_in, 1])
        x = tf.exp(-1 * tf.abs(x - means) / (1e-5 + sigma)) / (1e-5 + 2 * sigma)
        x_ml = x / tf.reduce_mean(x, axis=2, keep_dims=True)
        x = x_ml * x_expanded
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)