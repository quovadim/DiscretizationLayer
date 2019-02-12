import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from datasets import *
from models import *

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau, TensorBoard

import argparse

parser = argparse.ArgumentParser(description='Train net')
parser.add_argument('model', type=str, help="Network name")
parser.add_argument('dataset', type=str, help="Dataset name")

args = parser.parse_args()

model_name = args.model
if model_name not in models.keys():
    print 'No model', model_name, 'found'
    exit(-1)

dataset_name = args.dataset
if dataset_name not in datasets.keys():
    print 'No dataset', dataset_name, 'found'
    exit(-1)

dataset = datasets[dataset_name](rows=200000)

model, model_data = models[model_name](dataset.input_shape, dataset.output_shape, 256)

X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, train_size=0.8)

path = './' + dataset_name + '_' + model_name

if os.path.exists(path):
    os.system('rm -rf ' + path)

callbacks = [ReduceLROnPlateau(verbose=1, patience=10, min_lr=1e-6, factor=0.9),
             TensorBoard(path)]

model.fit(X_train, y_train, batch_size=1024, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)