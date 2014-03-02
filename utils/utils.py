from matplotlib import pyplot as plt
import numpy as np
from types import FunctionType


class HashableDict(dict):
    # http://code.activestate.com/recipes/414283-frozen-dictionaries/
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def imagesc(data, dest=None, grayscale=True, vmin=None, vmax=None):
    cmap = plt.cm.gray if grayscale else None
    if dest is None:
        fig = plt.figure(figsize=(7,4))
        show = plt.matshow(data, cmap=cmap, fignum=fig.number, vmin=vmin, vmax=vmax)
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
    else:
        show = dest.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        dest.axes.get_xaxis().set_visible(False)
        dest.axes.get_yaxis().set_visible(False)
    return show

def isunique(lst): return len(set(lst))==1

def isfunction(x): return isinstance(x, FunctionType)

def sigmoid(x):
    # from peter's rbm
    return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
    s = sigmoid(x)
    return s * (1. - s)

def tanh(x): return np.tanh(x)

def deriv_tanh(x): return 1. - np.power(tanh(x), 2)

def sumsq(x): return np.sum(np.power(x,2), axis=1)

def vec_to_arr(x): return x.reshape(1, x.size)

def between(x, lower=0, upper=1): return x > lower and x < upper
