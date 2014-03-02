from itertools import islice
import numpy as np


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks. If
    the last block is too small, returns a truncated block.

    e.g. grouper('ABCDEFG', 3) --> ABC DEF G

    From http://stackoverflow.com/a/8991553/230523
    """
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

def gen_minibatch_idx(n, n_per_mb):
    """
    Returns an iterator. Each step contains a tuple of
    indices of length N_PER_MB (though the last might be
    truncated). N is an int, defining the maximum index.
    """
    assert isinstance(n, int)
    all_idx = np.arange(n)
    np.random.shuffle(all_idx)
    for idx in grouper(all_idx, n_per_mb):
        yield idx

def gen_minibatch(arr1, arr2=None, n_in_mb=None):
    """
    e.g.
        from deprofundis.utils.minibatch import gen_minibatch
        arr1 = np.array([[10,20,30,40,50,60,70], [100,200,300,400,500,600,700], [1000,2000,3000,4000,5000,6000,7000]]).T
        arr2 = arr1 + 1
        for mb in gen_minibatch(arr1, arr2, 3):
            print mb
    """
    # if they fed in 2 args, switch the 2nd and 3rd
    if arr2 is not None and n_in_mb is None and isinstance(arr2, int):
        n_in_mb, arr2 = arr2, n_in_mb
    n = len(arr1)
    if arr2 is None:
        for idx in gen_minibatch_idx(n, n_in_mb):
            yield arr1[idx,:]
    else:
        assert arr1.shape[0] == arr2.shape[0]
        for idx in gen_minibatch_idx(n, n_in_mb):
            yield arr1[idx,:], arr2[idx,:]
