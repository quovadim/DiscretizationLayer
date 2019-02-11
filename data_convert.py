import numpy as np
import pandas as pd
from tqdm import tqdm


def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in tqdm(np.ndindex(shp), total=len(a)):
        if np.random.uniform(0, 1) < 0.5:
            np.random.shuffle(b[ndx])
    return

higgs_dataset = {'file': './HIGGS.csv', 'input': 28, 'y label': '0'}
susy_dataset = {'file': './SUSY.csv', 'input': 18, 'y label': '0'}
htru_dataset = {'file': './HTRU_2.csv', 'input': 8, 'y label': '8'}

cdataset = susy_dataset

classes = 2

print 'Reading...'
header = [str(i) for i in range(cdataset['input'] + 1)]
X = pd.read_csv(cdataset['file'], nrows=5000000, names=header)
print X.head()

y = X[cdataset['y label']].values.reshape(-1, 1)
X = X.drop(columns=[cdataset['y label']]).values

disarrange(X, axis=1)

total = np.concatenate([y, X], axis=1)

pd.DataFrame(total).to_csv('SUSY4.csv', index=False)