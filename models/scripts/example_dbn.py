from models.dbn import Layer, DBN
from models.distribution import *
from models.optimizer import *
from models.sampler import *
from models.rbm import *
from utils.utils import prepare_batches
from data.mnist.path import *
import pandas as pandas

SIZE_BATCH = 10
EPOCHS = 1
SIZE_HIDDEN_2 = 300
SIZE_HIDDEN_1 = 500
SIZE_VISIBLE = 784

# load binary mnist sample dataset
dataset = pandas.read_csv(MNIST_SAMPLE_BIN, delimiter=',', dtype=int, header=None)
# leave the first column out since it contains the labels
dataset = dataset.values[:, 1:]
# compute batch set
idx = prepare_batches(len(dataset), SIZE_BATCH)

# setup layer no. 1
bernoulli_1 = Bernoulli(SIZE_VISIBLE, SIZE_HIDDEN_1)
gibbs_1 = BlockGibbsSampler(bernoulli_1, sampling_steps=1)
sgd_1 = SGD(bernoulli_1)
rbm_1 = RBM(bernoulli_1, gibbs_1, sgd_1)

layer_1 = Layer(rbm_1)

# setup layer no. 2
bernoulli_2 = Bernoulli(SIZE_HIDDEN_1, SIZE_HIDDEN_2)
gibbs_2 = BlockGibbsSampler(bernoulli_2, sampling_steps=1)
sgd_2 = SGD(bernoulli_2)
rbm_2 = RBM(bernoulli_2, gibbs_2, sgd_2)

layer_2 = Layer(rbm_2)

# Create DBN
layers = [layer_1, layer_2]
dbn = DBN(layers)

# Train DBN
dbn.greedy_train(dataset)
dbn.backfit(dataset)