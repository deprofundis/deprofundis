from matplotlib import pyplot as plt
import numpy as np
from types import FunctionType


class HashableDict(dict):
    # http://code.activestate.com/recipes/414283-frozen-dictionaries/
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def grab_minibatch(patterns, n_in_minibatch):
    assert len(patterns.shape) == 2
    ix = np.arange(len(patterns))
    np.random.shuffle(ix)
    ix = ix[:n_in_minibatch]
    return patterns[ix,:]


def prepare_batches(len_data,len_batch):
    """
    Computes the start and stop indexes for a batch of data.
    
    @param len_data: Length of the entire data set
    @param len_batch: Length of a single batch
    @return: An array that contains start and stop indexes
    """
    # compute number of batches
    num_batch = len_data / len_batch
    # compute minimum index length
    idx_list = np.arange(num_batch * len_batch)
    # shape to batch sizes
    idx_list = np.reshape(idx_list, (num_batch, len_batch))
    idx_list = np.array([idx_list[:,0], idx_list[:,-1]])
    idx_list = idx_list.T
    # take the first and the last column to have a start and a stop column
    mod = len_data % float(len_batch)
    if mod > 0:
        low_index = idx_list[-1,0] + 1
        high_index = (len_data-1)
        idx_list = np.vstack([idx_list, [low_index, high_index]])
    # shuffle our indizes
    np.random.shuffle(idx_list)
    return idx_list
    
def prepare_frames(len_data, len_chunk, len_batch):
    """
    Returns a list of batches where each batch contains a list of tuples defining 
    start/end point of a chunk respectively
    
    Note: Each chunk represents a sliding window which is moved by 1 data point (frame) every time
    
    @param len_data: Length of the entire data set
    @param len_chunk: Length of a single frame (eq. number of data points per frame)
    @param len_batch: Length of a single batch
    """
    # isolate chunks of length N+1/len_frame    
    start_idx_list = np.arange(len_data - len_chunk + 1)
    stop_idx_list = np.arange(len_chunk, len_data + 1)
    # create start/end idx tuples
    idx_list = zip(start_idx_list, stop_idx_list) 
    # shuffle chunks
    np.random.shuffle(idx_list)
    # create batches of size len_batch
    batch_idx_list = [idx[i:i+len_batch] for i in xrange(0, len(idx), len_batch)]
    
    return batch_idx_list

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
