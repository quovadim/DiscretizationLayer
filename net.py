import tensorflow as tf


def make_dense_net(input, neurons, input_shape, output_shape, is_training):
    bn = tf.layers.batch_normalization(input)
    dense = tf.layers.dense(bn, output_shape, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    logits = tf.layers.dense(dropout, 2)

    return logits, None, bn

def make_disc_net(input, neurons, input_shape, output_shape, is_training):
    bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((input_shape, output_shape), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='temp')
    x = tf.expand_dims(bn, -1)
    x = tf.tile(x, [1, 1, output_shape])
    x = tf.square(disc_bias) + -1 * tf.square(x - disc_means) * tf.square(disc_sigma)
    x = x / (1e-7 + tf.abs(disc_temperature))
    x_ml = tf.nn.softmax(x)
    x = tf.layers.flatten(x_ml)
    #dropout = tf.layers.dropout(x, 0.2, training=is_training)
    dense = tf.layers.dense(x, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)


    logits = tf.layers.dense(dropout, 2)

    return logits, x_ml, bn

def make_disc_net_small(input, neurons, input_shape, output_shape, is_training):
    bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((1, output_shape), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='temp')
    x = tf.expand_dims(bn, -1)
    x = tf.tile(x, [1, 1, output_shape])
    means = tf.tile(disc_means, [input_shape, 1])
    bias = tf.tile(disc_bias, [input_shape, 1])
    sigma = tf.tile(disc_sigma, [input_shape, 1])
    x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
    x = x / (1e-7 + tf.abs(disc_temperature))
    x_ml = tf.nn.softmax(x)
    x = tf.layers.flatten(x_ml)
    #dropout = tf.layers.dropout(x, 0.2, training=is_training)
    dense = tf.layers.dense(x, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)

    logits = tf.layers.dense(dropout, 2)

    return logits, x_ml, bn

def weird_layer(input, dimension_in, dimension_out):
    bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((1, dimension_out), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((1, dimension_out), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((1, dimension_out), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((1, dimension_out), mean=1, stddev=0.1), name='temp')
    x_expanded = tf.expand_dims(bn, -1)
    i_expanded = tf.expand_dims(input, -1)
    x = tf.tile(x_expanded, [1, 1, dimension_out])
    means = tf.tile(disc_means, [dimension_in, 1])
    bias = tf.tile(disc_bias, [dimension_in, 1])
    sigma = tf.tile(disc_sigma, [dimension_in, 1])
    x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
    x = x / (1e-7 + tf.abs(disc_temperature))
    x_ml = tf.nn.softmax(x)
    x = x_ml * i_expanded
    x = tf.reduce_sum(x, axis=1)
    return tf.layers.batch_normalization(x)

def make_super_dense_net(input, neurons, input_shape, output_shape, is_training):
    dense = weird_layer(input, input_shape, output_shape)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = weird_layer(dropout, output_shape, neurons)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = weird_layer(dropout, neurons, neurons / 4)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = weird_layer(dropout, neurons / 4, neurons / 16)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    logits = tf.layers.dense(dropout, 2)

    return logits, None, None

def make_disc_net_weird(input, neurons, input_shape, output_shape, is_training):
    #bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((1, output_shape), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='temp')
    x_expanded = tf.expand_dims(input, -1)
    x = tf.tile(x_expanded, [1, 1, output_shape])
    means = tf.tile(disc_means, [input_shape, 1])
    bias = tf.tile(disc_bias, [input_shape, 1])
    sigma = tf.tile(disc_sigma, [input_shape, 1])
    x = tf.square(bias) + -1 * tf.square(x - means) * tf.square(sigma)
    x = x / (1e-7 + tf.abs(disc_temperature))
    x_ml = tf.nn.softmax(x)
    x = x_ml * x_expanded
    x = tf.reduce_sum(x, axis=1)
    dense = tf.layers.dense(x, neurons, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)

    logits = tf.layers.dense(dropout, 2)

    return logits, x_ml, input

def make_disc_net_weird_full(input, neurons, input_shape, output_shape, is_training):
    bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((input_shape, output_shape), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='temp')
    x_expanded = tf.expand_dims(bn, -1)
    x = tf.tile(x_expanded, [1, 1, output_shape])
    x = tf.square(disc_bias) + -1 * tf.square(x - disc_means) * tf.square(disc_sigma)
    x = x / (1e-7 + tf.abs(disc_temperature))
    x_ml = tf.nn.softmax(x)
    x = x_ml * x_expanded
    x = tf.reduce_sum(x, axis=1)
    dense = tf.layers.dense(x, neurons, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)

    logits = tf.layers.dense(dropout, 2)

    return logits, x_ml, bn

def make_laplace_net(input, neurons, input_shape, output_shape, is_training):
    bn = tf.layers.batch_normalization(input)
    disc_means = tf.Variable(tf.random_uniform((input_shape, output_shape), minval=-2, maxval=2), name='means')
    disc_bias = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='bias')
    disc_sigma = tf.Variable(tf.random_normal((input_shape, output_shape), mean=1, stddev=0.1), name='sigma')
    disc_temperature = tf.Variable(tf.random_normal((1, output_shape), mean=1, stddev=0.1), name='temp')
    x = tf.expand_dims(bn, -1)
    x = tf.tile(x, [1, 1, output_shape])
    x = -1 * tf.abs(x - disc_means) / (1 + tf.abs(disc_sigma))
    x = tf.exp(x) / (1 + 2 * tf.abs(disc_sigma))
    #x = x / (1e-7 + tf.abs(disc_temperature))
    #x_ml = tf.nn.softmax(x)
    x_ml = x / tf.reduce_sum(x, axis=2, keepdims=True)
    #x = x / (1e-7 + tf.abs(disc_temperature))
    #x_ml = tf.nn.softmax(x)
    x = tf.layers.flatten(x_ml)
    #dropout = tf.layers.dropout(x, 0.2, training=is_training)
    dense = tf.layers.dense(x, neurons / 4, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)
    dense = tf.layers.dense(dropout, neurons / 16, tf.nn.elu)
    dropout = tf.layers.dropout(dense, 0.2, training=is_training)

    logits = tf.layers.dense(dropout, 2)

    return logits, x_ml, bn