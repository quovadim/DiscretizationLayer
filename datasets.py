from abstract_dataset import AbstractDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from keras.utils import to_categorical


def disarrange(a, axis=-1, seed=None):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    np.random.seed(seed)
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in tqdm(np.ndindex(shp), total=len(a)):
        if np.random.uniform(0, 1) < 0.5:
            np.random.shuffle(b[ndx])
    return


class SusyDataset(AbstractDataset):
    # link : https://archive.ics.uci.edu/ml/datasets/SUSY
    def __init__(self, rows=None):
        super(SusyDataset, self).__init__()
        self.path = './data/SUSY.csv'
        self.rows = rows
        self.load()
        self.input_shape = self.X.shape[1]
        self.output_shape = self.y.shape[1]

    def load(self, filename=None):
        data = pd.read_csv(self.path, nrows=self.rows, names=range(19), verbose=True)
        self.y = to_categorical(data[0].values, num_classes=2)
        self.X = data.drop(labels=[0], axis=1).values

    def save(self, filename):
        pd.DataFrame(np.concatenate([self.y, self.X], axis=1)).to_csv(filename, index=False, header=None)


class SusyShuffledDataset(AbstractDataset):
    # link : https://archive.ics.uci.edu/ml/datasets/SUSY
    def __init__(self, rows=None):
        super(SusyShuffledDataset, self).__init__()
        self.path = './data/SUSY.csv'
        self.rows = rows
        self.load()
        self.input_shape = self.X.shape[1]
        self.output_shape = self.y.shape[1]

    def load(self, filename=None):
        print 'Reading...'
        data = pd.read_csv(self.path, nrows=self.rows, verbose=True)
        print '...Done'
        self.y = to_categorical(data['0'].values, num_classes=2)
        self.X = data.drop(labels=['0'], axis=1).values
        print 'Shuffling'
        disarrange(self.X, axis=1, seed=29)
        print 'Done'

    def save(self, filename):
        pd.DataFrame(np.concatenate([self.y, self.X], axis=1)).to_csv(filename, index=False, header=None)


datasets = {'susy': SusyDataset,
            'susy_shuffled': SusyShuffledDataset}
