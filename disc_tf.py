import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle

from net import *

from sklearn.model_selection import train_test_split

import sys

import argparse

parser = argparse.ArgumentParser(description='Train net')
parser.add_argument('-d', '--dense', action='store_true', help="Train dense")
parser.add_argument('-e', '--entropy_constraint', type=float, default=0, help="Regularization coef")
parser.add_argument('-a', '--entropy_balance', type=float, default=0.5, help="Balance coef")
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('-m', '--mode', type=str, default='full', help="Mode")

args = parser.parse_args()

tf.random.set_random_seed(27)
np.random.seed(27)


def visualize(mean, sigma, bias, temperature, x, data):
    x = np.concatenate([x.reshape(-1, 1)] * len(mean), axis=1)
    plt.clf()
    y = -1 * np.abs(x - means) / (1 + np.abs(sigma))
    y = np.exp(y) / (1 + 2 * np.abs(sigma))
    y = y / (1e-7 + np.abs(temperature))
    y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
    #y = np.square(bias) - np.square(x - mean) * np.square(sigma) / np.abs((1e-7 + temperature))
    #y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
    #y = np.max(y, axis=1)
    plt.hist(data, normed=True, bins=1000)
    #plt.plot(x[:, 0], y)

    for i in range(len(mean)):
        plt.plot(x[:, 0], y[:, i], alpha=0.6)
    plt.savefig('./' + str(epoch) + '.png')


higgs_dataset = {'file': './HIGGS2.csv', 'input': 28, 'y label': '0'}
susy_dataset = {'file': './SUSY2.csv', 'input': 18, 'y label': '0'}
htru_dataset = {'file': './HTRU_22.csv', 'input': 8, 'y label': '0'}

cdataset = susy_dataset

classes = 2

print 'Reading...'
header = [str(i) for i in range(cdataset['input'] + 1)]
X = pd.read_csv(cdataset['file'], nrows=100000)
print X.head()

y = X[cdataset['y label']].values
y = y.astype(np.int32)
y_r = np.zeros((len(y), 2))
for i in range(len(y)):
    y_r[i, y[i]] = 1
y = np.asarray(y_r)
X = X.drop(columns=[cdataset['y label']]).values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27, train_size=0.85)

print 'X', X.shape, 'y', y.shape

input_shape = X.shape[1]
layer_steps = 1 * input_shape

input_layer = tf.placeholder(tf.float32, [None, input_shape])
y_ = tf.placeholder(tf.float32, [None, 2])

dense = args.dense

input_dim = X.shape[1]
map_dim = 2 * input_dim
alpha = 0.5
p = args.entropy_constraint
a = args.entropy_balance

mode = args.mode

if dense:
    dense_logits, xml, bn = make_dense_net(input_layer, input_dim * map_dim, input_dim, map_dim)
else:
    if mode == 'small':
        dense_logits, xml, bn = make_disc_net_small(input_layer, input_dim * map_dim, input_dim, map_dim)
    if mode == 'full':
        dense_logits, xml, bn = make_disc_net(input_layer, input_dim * map_dim, input_dim, map_dim)
    if mode == 'laplace':
        dense_logits, xml, bn = make_laplace_net(input_layer, input_dim * map_dim, input_dim, map_dim)
    if mode =='weird':
        dense_logits, xml, bn = make_disc_net_weird(input_layer, input_dim * map_dim, input_dim, map_dim)
    if mode =='weird_full':
        dense_logits, xml, bn = make_disc_net_weird_full(input_layer, input_dim * map_dim, input_dim, map_dim)
if dense:
    loss_crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_logits, labels=y_))
    loss = loss_crossentropy
else:
    loss_crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_logits, labels=y_))
    if p > 1e-7:
        loss_entropy_mean = tf.reduce_mean(xml, axis=0)
        loss_entropy_mean = tf.reduce_sum(-1 * loss_entropy_mean * tf.log(1e-5 + loss_entropy_mean), axis=1)
        loss_entropy_mean = -1 * tf.log(1e-5 + 1./ map_dim) - tf.reduce_mean(loss_entropy_mean)
        loss_mean_entropy = -1 * xml * tf.log(1e-5 + xml)
        loss_mean_entropy = tf.reduce_sum(loss_mean_entropy, axis=2)
        loss_mean_entropy = tf.reduce_mean(loss_mean_entropy)
        loss = alpha * loss_entropy_mean + (1 - alpha) * loss_mean_entropy
        loss = p * loss + loss_crossentropy
    else:
        loss = loss_crossentropy

training_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(dense_logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

batch_size = 1024

feature = 0

history_train = {'acc': [], 'loss': []}
history_test = {'acc': [], 'loss': []}

for epoch in range(2500):
    feed_train = {input_layer: X_train, y_: y_train}
    feed_test = {input_layer: X_test, y_: y_test}
    loss_hist = []
    acc_hist = []
    if not dense:
        pred = sess.run(xml, feed_dict=feed_train)[0][feature]
        means = tf.get_default_graph().get_tensor_by_name("means:0").eval(session=sess)[feature]
        bias = tf.get_default_graph().get_tensor_by_name("bias:0").eval(session=sess)[feature]
        sigma = tf.get_default_graph().get_tensor_by_name("sigma:0").eval(session=sess)[feature]
        temp = tf.get_default_graph().get_tensor_by_name("temp:0").eval(session=sess)[0][feature]

        pstr = ['{:6.3f}'] * len(means)
        pstr = ' '.join(pstr)
        print 'Pred  :', pstr.format(*pred)
        print 'Mean  :', pstr.format(*means)
        print 'Bias  :', pstr.format(*bias)
        print 'Sigma :', pstr.format(*sigma)
        print 'Temp  :', temp
        data = sess.run(bn, feed_train)
        #visualize(means, bias, sigma, temp, np.linspace(-3, 3).reshape(-1, 1), data[:, feature].reshape(-1, 1))

    sys.stderr.write('------------------------\n')

    for i in range(0, len(X_train), batch_size):
        s = i
        e = min(len(X_train), i + batch_size)
        feed_train_batch = {input_layer: X_train[s:e], y_: y_train[s:e]}
        sess.run(training_step, feed_dict=feed_train_batch)
        train_loss = sess.run(loss_crossentropy, feed_dict=feed_train_batch)
        accuracy_train = (sess.run(accuracy, feed_dict=feed_train_batch))
        loss_hist.append(train_loss)
        acc_hist.append(accuracy_train)
        print_string = '\rEpoch {:5d} Samples {:10d} | Train : loss {:8.5f} acc {:8.5f}'
        sys.stderr.write(print_string.format(epoch, i, np.mean(loss_hist), np.mean(acc_hist)))
    train_loss = sess.run(loss_crossentropy, feed_dict=feed_train)
    test_loss = sess.run(loss_crossentropy, feed_dict=feed_test)
    accuracy_train = (sess.run(accuracy, feed_dict=feed_train))
    accuracy_test = (sess.run(accuracy, feed_dict=feed_test))

    history_train['acc'].append(accuracy_train)
    history_train['loss'].append(train_loss)

    history_test['acc'].append(accuracy_test)
    history_test['loss'].append(test_loss)

    suffix = 'dense'
    if not dense:
        suffix = 'disc_' + mode

    if epoch % 10 == 0:
        with open('dump_' + suffix + '.pkl', 'w') as f:
            pickle.dump([history_train, history_test], f)

    print_string = '\rEpoch {:5d} Samples {:10d} | Train : loss {:8.5f} acc {:8.5f} | Test : loss {:8.5f} acc {:8.5f}\n'

    sys.stderr.write(print_string.format(epoch, len(X_train), train_loss, accuracy_train, test_loss, accuracy_test))

    sys.stderr.write('------------------------\n')

print ''