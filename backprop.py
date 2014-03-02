from ipdb import set_trace as pause
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle

from base import create_mnist_patternsets, Minibatch2, Network, Patternset
from plotting import plot_errors
from utils.utils import deriv_sigmoid, deriv_tanh, imagesc, sigmoid, sumsq, tanh, vec_to_arr


class BackpropNetwork(Network):
    def __init__(self, layersizes, lrate=0.01, momentum=0.9, plot=True):
        assert len(layersizes) >= 3 # incl input & output
        self.layersizes = layersizes
        self.w, self.d_w = self.init_weights(scale=1), self.init_weights(0)
        self.b, self.d_b = self.init_biases(), self.init_biases()
        self.n_l = len(self.layersizes)
        self.momentum = momentum
        self.lrate = lrate

        self.plot = plot
        self.fignum_errors = 1
        self.fignum_layers = 2
        self.fignum_weights_01 = 3
        self.fignum_dweights_01 = 4
        self.fignum_weights_12 = 5
        self.fignum_dweights_12 = 6
        self.fignum_biases = 7
        self.fignum_dbiases = 8

    def init_weights(self, scale=0.01):
        return [np.random.normal(size=(n1,n2), scale=scale) if scale else np.zeros((n1,n2))
                for n1,n2 in zip(self.layersizes, self.layersizes[1:])]

    def init_biases(self): return [np.zeros((n,)) for n in self.layersizes]

    def test_trial(self, act0, tgt):
        inps, acts = self.propagate_fwd_all(act0)
        act_k = acts[-1]
        err = self.report_error(act_k, tgt)
        return err, act_k

    def propagate_fwd_all(self, act0):
        inps, acts = [act0]*self.n_l, [act0]*self.n_l
        for l in range(self.n_l-1):
            inps[l+1], acts[l+1] = self.propagate_fwd(l, acts[l])
        return inps, acts

    def propagate_fwd(self, lowlayer_idx, act):
        w, b = self.w[lowlayer_idx], self.b[lowlayer_idx+1]
        return super(BackpropNetwork, self).propagate_fwd(act, w, b)

    def report_error(self, act_k, tgt): return np.sum(np.power(tgt - act_k, 2), axis=1)

    def learn_trial(self, act0, target):
        d_w, d_b = self.delta_weights(act0, target)
        for cur, new, old in zip(self.w, d_w, self.d_w): cur += new + self.momentum*old
        for cur, new, old in zip(self.b, d_b, self.d_b): cur += new + self.momentum*old
        self.d_w, self.d_b = d_w, d_b

    def delta_weights(self, act0, tgt):
        d_w, d_b = self.init_weights(0), self.init_biases()
        inps, acts = self.propagate_fwd_all(act0)
        j, k = self.n_l-2, self.n_l-1
        sens = [None] * self.n_l # pre-initialize SENSitivity arrays
        d_w[j], d_b[k], sens[k] = self.delta_w_jk(tgt-acts[k], inps[k], acts[j])
        for i in reversed(range(self.n_l-2)):
            j, k = i+1, i+2
            d_w[i], d_b[j], sens[j] = self.delta_w_ij(inps[j], self.w[j], sens[k], acts[i])
        return d_w, d_b

    def delta_w_jk(self, err_k, inp_k, act_j):
        """
        Changing W_JK, i.e. the weights from penultimate
        (hidden) layer J to uppermost (output) layer K.
        """
        sens_k = err_k * self.deriv_act_fn(inp_k)
        d_w_jk = self.lrate * np.dot(act_j.T, sens_k)
        d_b_k = np.mean(self.lrate * sens_k, axis=0)
        return d_w_jk, d_b_k, sens_k

    def delta_w_ij(self, inp_j, w_jk, sens_k, act_i):
        """
        Changing W_IJ, i.e. the weights from (input or
        hidden) layer I to next (hidden) layer J.
        """
        sens_j = self.deriv_act_fn(inp_j) * np.dot(w_jk, sens_k.T).T
        d_w_ij = self.lrate * np.dot(act_i.T, sens_j)
        d_b_j = np.mean(self.lrate * sens_j, axis=0)
        return d_w_ij, d_b_j, sens_j

    def act_fn(self, x): return sigmoid(x)
    def deriv_act_fn(self, x): return deriv_sigmoid(x)

#     def act_fn(self, x): return tanh(x)
#     def deriv_act_fn(self, x): return deriv_tanh(x)
    
    def err_fn(self, actual, desired): return np.pow(actual - desired, 2)

    def plot_errors(self, train_errors):
        if not self.plot: return
        return plot_errors(train_errors, fignum=self.fignum_errors)


def arr_str(arr): return np.array_str(arr, precision=2)

def test_epoch(net, acts0, targets, epochnum=-1, verbose=True):
    errors, outs = net.test_trial(acts0, targets)
    if verbose:
        for act0, target, out, error in zip(acts0, targets, outs, errors):
            print '%s, %s -> %s, err = %.2f' % \
                (arr_str(act0), arr_str(target), arr_str(out), error)
    mean_error = np.mean(errors)
    print '%i) err = %.2f' % (epochnum, mean_error)
    return errors, mean_error

def xor(*args, **kwargs):
    data = [[[0,0], [0]],
            [[0,1], [1]],
            [[1,0], [1]],
            [[1,1], [0]]
            ]
    iset = Patternset([vec_to_arr(np.array(a)) for a,t in data])
    oset = Patternset([vec_to_arr(np.array(t)) for a,t in data])
    net = BackpropNetwork([2, 2, 1], *args, **kwargs)
    return net, iset, oset

def autoencoder(n_inp_out, nhiddens, *args, **kwargs):
    layersizes = [n_inp_out] + nhiddens + [n_inp_out]
    net = BackpropNetwork(layersizes, *args, **kwargs)
    return net

def rand_autoencoder(nhiddens=None, *args, **kwargs):
    n_inp_out = 4
    npatterns = 10
    if nhiddens is None: nhiddens = [12]
    pset = Patternset([np.random.uniform(size=(1,n_inp_out)) for d in range(npatterns)])
    return autoencoder(n_inp_out, nhiddens=nhiddens, *args, **kwargs)

def mnist_autoencoder(nhiddens=None, *args, **kwargs):
    if nhiddens is None: nhiddens = [500]
    return autoencoder(784, nhiddens=nhiddens, *args, **kwargs)


if __name__ == "__main__":
    np.random.seed()
    # net, iset, oset = xor(lrate=0.35)
    # net, iset, oset = rand_autoencoder(nhiddens=[24, 12], lrate=0.1)
    train_pset, valid_pset, test_pset = create_mnist_patternsets(n_trainpatterns=1000, ravel=True, zero_to_negone=True)
    net = mnist_autoencoder(nhiddens=[200], lrate=0.01)
    train_iset, train_oset = train_pset, train_pset
    nEpochs = 20000 # 100000
    n_in_minibatch = min(len(train_iset), 20)
    report_every = 100 # 1000
    train_errors = []
    for e in range(nEpochs):
        act0, target = Minibatch2(train_iset, train_oset, n_in_minibatch).patterns
        # act0, target = act0.ravel(), target.ravel()
        net.learn_trial(act0, target)
        if not e % report_every:
            error, mean_error = test_epoch(net, train_iset.patterns, train_oset.patterns, False)
            train_errors.append(mean_error)
            net.plot_errors(train_errors)
            if mean_error < 0.01: break
    final_error, final_mean_error = test_epoch(net, train_iset.patterns, train_oset.patterns, True)
