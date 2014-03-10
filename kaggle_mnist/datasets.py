from ipdb import set_trace as pause
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.label import label_binarize

from ..utils.stopwatch import Stopwatch


class Dataset(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_kaggle_mnist_train(filen, nrows=None, zero_to_negone=False):
    """
    Reads in the Kaggle MNIST training dataset and returns X
    (input) and Y (label) data.

    Dataset should be a .csv or .csv.gz file
    (NOBSERVATIONS+1 x NDIMENSIONS+1), with first row =
    header, and first column = labels.
    """
    print 'Loading %s Kaggle MNIST train patterns from %s' % (str(nrows) if nrows else 'all', filen)
    t = Stopwatch()
    panda = pd.read_csv(filen, delimiter=',', dtype=int, header=None, nrows=nrows,
                        skiprows=1, compression=('gzip' if filen.endswith('.gz') else None))
    data = panda.values # numpy array
    x = data[:,1:]
    y_vec = data[:,0]

    assert x.shape[0] == y_vec.shape[0]
    assert x.shape[1] == 784
    assert len(y_vec.shape) == 1
    assert np.min(y_vec) == 0 and np.max(y_vec) == 9
    # turn labels from vector (with values from 0-9) to
    # NOBSERVATIONSx10 matrix, with a single 1 in each row (i.e. 1-vs-all)
    y = label_binarize(y_vec, classes=range(10))
    assert y.shape == (x.shape[0], 10)
    assert all(np.sum(y, axis=1) == 1)
    if zero_to_negone: x[x==0] = -1
    if nrows is not None: assert x.shape[0] == nrows
    print 'done: %r in %is' % (x.shape, t.finish(milli=False))
    return x, y


def load_kaggle_mnist_test(filen, nrows=None, zero_to_negone=False):
    """
    Reads in the Kaggle MNIST test dataset and returns X
    (input) data (but not the Y-label data, since we have to
    predict that).

    Dataset should be a .csv or .csv.gz file
    (NOBSERVATIONS+1 x NDIMENSIONS), with first row =
    header. No labels column.
    """
    print 'Loading %s Kaggle MNIST test patterns from %s' % (str(nrows) if nrows else 'all', filen)
    t = Stopwatch()
    panda = pd.read_csv(filen, delimiter=',', dtype=int, header=None, nrows=nrows,
                        skiprows=1, compression=('gzip' if filen.endswith('.gz') else None))
    data = panda.values # numpy array
    x = data
    assert x.shape[1] == 784
    if zero_to_negone: x[x==0] = -1
    if nrows is not None: assert x.shape[0] == nrows
    print 'done: %r in %is' % (x.shape, t.finish(milli=False))
    return x

def load_kaggle_mnist_both(train_filen, test_filen, train_n=None, test_n=None):
    trainvalid_x, trainvalid_y = load_kaggle_mnist_train(filen=train_filen, nrows=train_n)
    train_x, valid_x, train_y, valid_y = train_test_split(trainvalid_x, trainvalid_y,
                                                          test_size=0.1, random_state=0)
    test_x = load_kaggle_mnist_test(filen=test_filen, nrows=test_n)
    return train_x, train_y, valid_x, valid_y, test_x

