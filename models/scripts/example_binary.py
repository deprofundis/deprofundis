from models.sampler import BlockGibbsSampler
from models.distribution import Bernoulli
from models.optimizer import SGD
from models.rbm import RBM
from data.mnist.path import *
from utils.utils import prepare_batches
from matplotlib import pyplot

import pandas, numpy, time

SIZE_BATCH = 10
EPOCHS = 1
SIZE_HIDDEN = 500
SIZE_VISIBLE = 784

# load binary mnist sample dataset
dataset = pandas.read_csv(MNIST_TRAIN_BINARY, delimiter=',', dtype=int, header=None)
# leave the first column out since it contains the labels
dataset = dataset.values[:, 1:]
# compute batch set
idx = prepare_batches(len(dataset), SIZE_BATCH)

# load distribution
bernoulli = Bernoulli(SIZE_VISIBLE, SIZE_HIDDEN)
gibbs = BlockGibbsSampler(bernoulli, sampling_steps=1)
sgd = SGD(bernoulli)
rbm = RBM(bernoulli, gibbs, sgd)

y_axis = list()

pyplot.plot(y_axis)
pyplot.ylabel('update')
pyplot.xlabel('Batch')
pyplot.ion()
pyplot.show()

for epoch in range(EPOCHS):
    for b_idx in idx:
        batch = dataset[b_idx[0]:b_idx[1], :]
        d_weight_update, _, _ = rbm.train_batch(batch)
        rec_error = rbm.compute_reconstruction_error(batch)

        y_axis.append(rec_error)
        # make simple plot
        if len(y_axis) % 100 is 0:
            pyplot.plot(y_axis)
            pyplot.draw()
            time.sleep(0.1)
