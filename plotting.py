from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils.utils import imagesc


def plot_biases(v_bias, h_bias, fignum=None, ttl=None):
    vmin = None # min(map(min, [v_bias, h_bias]))
    vmax = None # max(map(max, [v_bias, h_bias]))
    fig = plt.figure(fignum)
    plt.clf()
    if ttl: fig.suptitle(ttl + '. range=%.2f to %.2f' % (vmin or float('NaN'), vmax or float('NaN')))
    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0]); im = imagesc(v_bias, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v bias'); fig.colorbar(im)
    ax = fig.add_subplot(gs[0,1]); im = imagesc(h_bias, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h bias'); fig.colorbar(im)
    plt.draw()


def plot_errors(train_errors, valid_errors, test_errors, fignum=None):
    plt.figure(figsize=(9,6), num=fignum)
    plt.clf()
    epochrange = range(len(train_errors))
    assert len(train_errors) == len(valid_errors)
    assert len(train_errors) == len(test_errors)
    train_line = plt.plot(train_errors, label='Train')
    valid_line = plt.plot(valid_errors, label='Valid')
    test_line = plt.plot(test_errors, label='Test')
    plt.legend()
    max_error = max(max(train_errors), max(test_errors), max(valid_errors))
    plt.ylim(ymin=0, ymax=max_error*1.1)
    plt.draw()


def plot_rbm_2layer(v_plus,
                    h_plus_inp, h_plus_prob, h_plus_state,
                    v_minus_inp, v_minus_prob, v_minus_state,
                    h_minus_inp, h_minus_prob, h_minus_state,
                    fignum=None, ttl=None):
    lmin, lmax = None, None
    fig = plt.figure(fignum)
    plt.clf()
    if ttl: fig.suptitle(ttl)
    gs = gridspec.GridSpec(16,2)
    # top left downwards
    ax = fig.add_subplot(gs[    0,0]); im = imagesc(h_plus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_state')
    ax = fig.add_subplot(gs[    1,0]); im = imagesc(h_plus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_prob')
    ax = fig.add_subplot(gs[    2,0]); im = imagesc(h_plus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_inp')
    ax = fig.add_subplot(gs[ 5: 8,0]); im = imagesc(v_plus, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_plus'); fig.colorbar(im) # , ticks=[lmin, lmax])
    # top right downwards
    ax = fig.add_subplot(gs[    0,1]); im = imagesc(h_minus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_state')
    ax = fig.add_subplot(gs[    1,1]); im = imagesc(h_minus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_prob')
    ax = fig.add_subplot(gs[    2,1]); im = imagesc(h_minus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_inp')
    ax = fig.add_subplot(gs[ 5: 8,1]); im = imagesc(v_minus_state*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_state'); fig.colorbar(im) # , ticks=[lmin, lmax])
    ax = fig.add_subplot(gs[ 9:12,1]); im = imagesc(v_minus_prob*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_prob'); fig.colorbar(im) # , ticks=[lmin, lmax])
    ax = fig.add_subplot(gs[13:16,1]); im = imagesc(v_minus_inp*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_inp'); fig.colorbar(im) # , ticks=[lmin, lmax])
    plt.draw()


