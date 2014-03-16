from models.rbm import RBM
from models.optimizer import SGD
from models.distribution import Distribution

import numpy as np
import copy as copy

class Layer(object):
    """
    Implement a single layer for a deep belief network
    """

    def __init__(self,rbm, lrate=0.001, weight_decay=0.0002, momentum=0.9):
        # assert that we have a restricted boltzman machine as input
        assert (isinstance(rbm, RBM))
        assert (lrate > 0)

        self.rbm = rbm
        self.lrate = lrate
        self.momentum = momentum
        self.wdecay = weight_decay

        # models
        self.rec_model = None
        self.gen_model = None

        # momentum
        self.d_gen_weights = np.zeros(self.rbm.model_distribution.weights.shape)
        self.d_rec_weights = np.zeros(self.rbm.model_distribution.weights.shape)
        self.d_bias_hidden = np.zeros(self.rbm.model_distribution.bias_hidden.shape)
        self.d_bias_visible = np.zeros(self.rbm.model_distribution.bias_visible.shape)

    def greedy_train(self, batch):
        self.rbm.train_batch(batch)

    def __model_init__(self):
        if self.rec_model is None:
            self.rec_model = copy.deepcopy(self.rbm.model_distribution)

        if self.gen_weights is None:
            self.gen_model = copy.deepcopy(self.rbm.model_distribution)

    def wake_pass(self, vis_inp):
        self.__model_init__()

        batch_size = len(vis_inp)

        # do one sampling pass
        prob_h, state_h = self.rec_model.state_h(vis_inp)
        prob_v, state_v = self.gen_model.state_v(state_h)

        # compute generative weight updates
        weight_update = (1. / batch_size) * np.dot((vis_inp - state_v).T, state_h)
        weight_update = self.lrate * weight_update \
                        - self.wdecay * self.gen_model.weights \
                        + self.momentum * self.d_gen_weights
        self.gen_model.weights += weight_update
        self.d_gen_weights = weight_update

        # compute bias updates
        visible_bias_update = np.mean((vis_inp - state_v))
        visible_bias_update = self.lrate * visible_bias_update \
                              - self.wdecay * self.gen_model.bias_visible \
                              + self.momentum * self.d_bias_visible
        self.gen_model.bias_visible += visible_bias_update
        self.d_bias_visible = visible_bias_update

        return prob_h, state_h


    def sleep_pass(self, hid_inp):
        self.__model_init__()

        batch_size = len(hid_inp)

        # do one sampling pass
        prob_v, state_v = self.gen_model.state_v(hid_inp)
        prob_h, state_h = self.rec_model.state_h(state_v)

        # compute recognition weight updates
        weight_update = (1. / batch_size) * np.dot(state_v.T, (hid_inp - state_h))
        weight_update = self.lrate * weight_update \
                        - self.wdecay * self.rec_model.weights \
                        + self.momentum * self.d_rec_weights
        self.rec_model.weights += weight_update
        self.d_rec_weights = weight_update

        # compute bias updates
        hidden_bias_update = np.mean((hid_inp - state_h))
        hidden_bias_update = self.lrate * hidden_bias_update \
                             - self.wdecay * self.rec_model.bias_hidden \
                              + self.momentum * self.d_bias_hidden
        self.rec_model.bias_hidden += hidden_bias_update
        self.d_bias_hidden = hidden_bias_update

        return prob_v, state_v

class DBN(object):

    def __init__(self):
        pass

    def greedy_train(self, batch):
        pass

    def backfit(self, batch):
        pass