from models.sampler import BlockGibbsSampler
from models.distribution import GaussianBinary
from models.optimizer import SGD
from models.rbm import RBM
from data.mnist.path import *
from utils.utils import prepare_batches
from matplotlib import pyplot
import sklearn.preprocessing as pre
import pandas, numpy, time

SIZE_BATCH = 10
EPOCHS = 10
SIZE_HIDDEN = 500
SIZE_VISIBLE = 784

# load binary mnist sample dataset
dataset = pandas.read_csv(MNIST_TRAIN, delimiter=',', dtype=numpy.float64, header=None)
# leave the first column out since it contains the labels
# dataset must be normalized to have unit variance by column (sigma_i == 1)
dataset = dataset.values[:,1:]
# std = numpy.std(dataset, axis=0)
# std = numpy.diag(std)
# rec = numpy.reciprocal(std)
# rec[rec == numpy.inf] = 0
# dataset = numpy.dot(dataset,rec)

# compute batch set
idx = prepare_batches(len(dataset), SIZE_BATCH)

# load distribution
gaussian = GaussianBinary(SIZE_VISIBLE, SIZE_HIDDEN)
gibbs = BlockGibbsSampler(gaussian, sampling_steps=1)
sgd = SGD(gaussian, learning_rate=0.005, weight_decay=0, momentum=0)
rbm = RBM(gaussian, gibbs, sgd)

pyplot.figure(1)
pyplot.ion()
pyplot.show()
vmin = numpy.min(dataset)
vmax = numpy.max(dataset)

for epoch in range(EPOCHS):
    for b_idx in idx:
        batch = dataset[b_idx[0]:b_idx[1], :]
        d_weight_update, _, _ = rbm.train_batch(batch)
        rec_probs, rec_state = rbm.reconstruct(batch,steps=10)

        pyplot.clf()
        img = numpy.reshape(rec_state[-1,:], newshape=(28,28))

        print "Max: " + str(numpy.max(img)) + " Min: " + str(numpy.min(img))

        pyplot.matshow(img, fignum=0, cmap=pyplot.cm.gray, vmin=vmin , vmax=vmax)
        pyplot.draw()
        time.sleep(0.1)

raw_input()