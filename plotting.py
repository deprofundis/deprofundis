# from ipdb import set_trace as pause
from math import ceil, sqrt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils.utils import imagesc


def setup_plot():
    plt.ion()
    plt.show()

def plot_biases(v_bias, h_bias, fignum=None, ttl=None):
    setup_plot()
    fig = plt.figure(figsize=(3,2), num=fignum)
    vmin = None # min(map(min, [v_bias, h_bias]))
    vmax = None # max(map(max, [v_bias, h_bias]))
    fig = plt.figure(fignum)
    plt.clf()
    if ttl: fig.suptitle(ttl + '. range=%.2f to %.2f' % (vmin or float('NaN'), vmax or float('NaN')))
    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0]); im = imagesc(v_bias, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v bias'); fig.colorbar(im)
    ax = fig.add_subplot(gs[0,1]); im = imagesc(h_bias, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h bias'); fig.colorbar(im)
    plt.draw()


def plot_errors(train_errors, valid_errors=None, test_errors=None, fignum=None):
    setup_plot()
    fig = plt.figure(figsize=(4,3), num=fignum)
    plt.clf()
    epochrange = range(len(train_errors))
    if valid_errors is not None: assert len(train_errors) == len(valid_errors)
    if test_errors is not None: assert len(train_errors) == len(test_errors)
    train_line = plt.plot(train_errors, label='Train')
    valid_line = plt.plot(valid_errors, label='Valid')
    test_line = plt.plot(test_errors, label='Test')
    plt.legend()
    max_error = max(max(train_errors),
                    max(valid_errors) if valid_errors else 0,
                    max(test_errors) if test_errors else 0)
    plt.ylim(ymin=0, ymax=max_error*1.1)
    plt.draw()

def plot_rbm_2layer(v_plus=None,
                    h_plus_inp=None, h_plus_prob=None, h_plus_state=None,
                    v_minus_inp=None, v_minus_prob=None, v_minus_state=None,
                    h_minus_inp=None, h_minus_prob=None, h_minus_state=None,
                    fignum=None, ttl=None):
    setup_plot()
    lmin, lmax = None, None
    fig = plt.figure(figsize=(5,7), num=fignum)
    plt.clf()
    if ttl: fig.suptitle(ttl)
    gs = gridspec.GridSpec(16,2)
    # top left downwards
    if h_plus_state is not None:  ax = fig.add_subplot(gs[    0,0]); im = imagesc(h_plus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_state')
    if h_plus_prob is not None:   ax = fig.add_subplot(gs[    1,0]); im = imagesc(h_plus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_prob')
    if h_plus_inp is not None:    ax = fig.add_subplot(gs[    2,0]); im = imagesc(h_plus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_inp')
    if v_plus is not None:        ax = fig.add_subplot(gs[ 5: 8,0]); im = imagesc(v_plus, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_plus'); fig.colorbar(im) # , ticks=[lmin, lmax])
    # top right downwards
    if h_minus_state is not None: ax = fig.add_subplot(gs[    0,1]); im = imagesc(h_minus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_state')
    if h_minus_prob is not None:  ax = fig.add_subplot(gs[    1,1]); im = imagesc(h_minus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_prob')
    if h_minus_inp is not None:   ax = fig.add_subplot(gs[    2,1]); im = imagesc(h_minus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_inp')
    if v_minus_state is not None: ax = fig.add_subplot(gs[ 5: 8,1]); im = imagesc(v_minus_state*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_state'); fig.colorbar(im) # , ticks=[lmin, lmax])
    if v_minus_prob is not None:  ax = fig.add_subplot(gs[ 9:12,1]); im = imagesc(v_minus_prob*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_prob'); fig.colorbar(im) # , ticks=[lmin, lmax])
    if v_minus_inp is not None:   ax = fig.add_subplot(gs[13:16,1]); im = imagesc(v_minus_inp*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_inp'); fig.colorbar(im) # , ticks=[lmin, lmax])
    plt.draw()

def plot_weights(w, v_shape, fignum=None, ttl=None):
    setup_plot()
    fig = plt.figure(figsize=(9,6), num=fignum)
    n_h = w.shape[1]
    vmin, vmax = min(w.ravel()), max(w.ravel())
    fig = plt.figure(fignum)
    plt.clf()
    if ttl: fig.suptitle(ttl + '. range=%.2f to %.2f' % (vmin, vmax))
    nsubplots = int(ceil(sqrt(n_h)))
    gs = gridspec.GridSpec(nsubplots, nsubplots)
    for hnum in range(n_h):
        x,y = divmod(hnum, nsubplots)
        ax = fig.add_subplot(gs[x,y])
        im = imagesc(w[:,hnum].reshape(v_shape), dest=ax, vmin=vmin, vmax=vmax)
        # ax.set_title('to H#%i' % hnum)
    fig.colorbar(im)
    plt.draw()

