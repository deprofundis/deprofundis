from ipdb import set_trace as pause
import numpy as np
from random import sample

from datasets import load_mnist
from utils.stopwatch import Stopwatch
from utils.utils import imagesc, isunique, vec_to_arr


class Network(object):
    def __repr__(self):
        return '%s' % (self.__class__.__name__)

    def init_weights(self):
        self.w = np.random.uniform(size=(self.n_v, self.n_h))

    def act_fn(self, x): return x # linear activation function by default, i.e. no transformation

    def propagate_fwd(self, act1, w12, b2):
        # W = (N_LOWER x N_UPPER), ACT2 = (NPATTERNS x N_UPPER), B2 = (N_UPPER,)
        inp2 = np.dot(act1, w12) + b2
        act2 = self.act_fn(inp2)
        return inp2, act2

    def propagate_back(self, act2, w12, b1):
        # W = (N_LOWER x N_UPPER), ACT2 = (NPATTERNS x N_UPPER), B1 = (N_LOWER,)
        inp1 = np.dot(act2, w12.T) + b1
        # return INP1 as (NPATTERNS x N_LOWER)
        act1 = self.act_fn(inp1)
        return inp1, act1

    def calculate_error(self, target): raise NotImplementedError

    def update_weights(self, target): raise NotImplementedError

    def test_trial(self): raise NotImplementedError

    def learn_trial(self): raise NotImplementedError


class Patternset(object):
    def __init__(self, patterns, shape=None):
        # PATTERNS (inputs) = list of (X x Y) numpy arrays
        #
        # check they're all the same shape
        assert isunique([i.shape for i in patterns])
        if shape is None:
            # if it's a (N,) vector, turn it into a (N,1) array
            if len(patterns[0].shape) == 1: patterns = [vec_to_arr(i) for i in patterns]
        else:
            patterns = [i.reshape(shape) for i in patterns]
        self.patterns = patterns
        self.shape = patterns[0].shape
        self.n = len(patterns)

    def __repr__(self):
        return '%s I(%ix%i)x%i' % (
            self.__class__.__name__,
            self.shape[0], self.shape[1], len(self.patterns))

    def get(self, p): return self.patterns[p].ravel()

    def getmulti(self, ps): return [self.patterns[p].ravel() for p in ps]

    def imshow(self, x, dest=None): imagesc(x.reshape(self.shape), dest=dest)

    def __len__(self): return len(self.patterns)


class Minibatch(object):
    # TODO sample without replacement
    def __init__(self, pset, n):
        self.pset = pset
        self.n = n
        self.patterns = np.array(self.getmulti(sample(range(len(self.pset)), self.n)))
        
    def getmulti(self, idx): return self.pset.getmulti(idx)

class Minibatch2(object):
    def __init__(self, iset, oset, n):
        self.iset = iset
        self.oset = oset
        self.n = n
        assert iset.shape[0] == oset.shape[0] # total number of patterns
        idx = sample(range(len(self.iset)), self.n)
        self.ipatterns, self.opatterns = [np.array(x) for x in self.getmulti(idx)]
        self.patterns = self.ipatterns, self.opatterns

    def getmulti(self, idx): return self.iset.getmulti(idx), self.oset.getmulti(idx)


def create_mnist_patternsets(n_trainpatterns=50000, n_validpatterns=10000, ravel=False, zero_to_negone=False):
    def load_mnist_patternset(filen, npatterns=None):
        print 'Loading %s MNIST patterns from %s' % (str(npatterns) if npatterns else 'all', filen)
        t = Stopwatch()
        mnist_ds = load_mnist(filen=filen, nrows=npatterns)
        if zero_to_negone: mnist_ds.X[mnist_ds.X==0] = -1
        if npatterns is not None: assert mnist_ds.X.shape[0] == npatterns
        shape = (1,784) if ravel else (28,28)
        pset = Patternset(mnist_ds.X, shape=shape)
        print 'done (%i x %i) in %is' % (mnist_ds.X.shape[0], mnist_ds.X.shape[1], t.finish(milli=False))
        return pset

    train_filen = '../data/mnist_train.csv.gz'
    test_filen = '../data/mnist_test.csv.gz'
    n_trainvalidpatterns = n_trainpatterns + n_validpatterns
    train_valid_data = load_mnist_patternset(train_filen, npatterns=n_trainvalidpatterns)
    test_data = load_mnist_patternset(test_filen)
    train_data = Patternset(train_valid_data.patterns[:n_trainpatterns])
    valid_data = Patternset(train_valid_data.patterns[n_trainpatterns:n_trainvalidpatterns])
    assert train_data.n == 50000 if n_trainpatterns is None else n_trainpatterns
    assert valid_data.n == 10000 if n_validpatterns is None else n_validpatterns
    assert test_data.n == 10000
    return train_data, valid_data, test_data
