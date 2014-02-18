import numpy as np
import pandas as pd
from ipdb import set_trace as pause


class Dataset(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_mnist(filen='../data/mnist_train.csv.gz', nrows=None):
    """
    Reads in the MNIST dataset and returns a Dataset object which holds the data 
    along with metadata from the files specified
    """
    panda = pd.read_csv(filen, delimiter=',', dtype=int, header=None, nrows=nrows,
                    compression=('gzip' if filen.endswith('.gz') else None))
    data = panda.values # numpy array
    return Dataset(X=data, name='mnist', filen=filen)
