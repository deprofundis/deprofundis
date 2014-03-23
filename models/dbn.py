from models.rbm import RBM
from utils import utils
from models.optimizer import SGD
from models.distribution import Distribution

import numpy as np
import copy as copy
import plotting
import time


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

        if self.gen_model is None:
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

    def propagate_fwd(self, vis_inp):
        if self.rec_model is None:
            # we are still in the greedy training
            return self.rbm.model_distribution.state_h(vis_inp)
        else:
            # Greedy layerwise trianing is done and we are in the backfitting phase
            return self.rec_model.state_h(vis_inp)


class DBN(object):
    """
    Class that represents a Deep Belief network with arbitrary distributions
    """
    def __init__(self, dbn_layers):
        """
        @param dbn_layers: Contains all the different layer of the rbm, whereas the layer at index 0 is the start layer
        """
        self.dbn_layers = self.__check_layer_fit__(dbn_layers)
    def __check_layer_fit__(self, dbn_layers):
        """
        Function checks that layers are correctly sized which means that the number of hidden units of a layer has to
        be the same as the number of visible units in the consecutive layer.
        @param dbn_layers: list that contains all the
        @return: Returns the list with the all the layers if passed through correctly. If sizes do not fit, an
        Assertation Error will be thrown
        """
        assert (len(dbn_layers) > 1)

        for index,layer in enumerate(dbn_layers):
            if index < (len(dbn_layers) - 1):
                cons_layer = dbn_layers[index + 1]
                hidden_pres = layer.rbm.model_distribution.size_hidden
                visible_cons = cons_layer.rbm.model_distribution.size_visible
                # assert that layers fit into each other
                assert (hidden_pres == visible_cons)

        return dbn_layers

    def greedy_train(self, data, batch_size=10, epochs=1):
        idx = utils.prepare_batches(len(data), batch_size)
        for index,layer in enumerate(self.dbn_layers):
            for epoch in range(epochs):
                for b_idx in idx:
                    batch = data[b_idx[0]:b_idx[1], :]
                    # propagate the batch through the already trained
                    for prop in range(index):
                        prop_layer = self.dbn_layers[prop]
                        batch,_ = prop_layer.propagate_fwd(batch)

                    # train RBM in a single layer
                    layer.greedy_train(batch)

            # initialize the consecutive layer with the biases
            if index < len(self.dbn_layers) - 1:
                cons_layer = self.dbn_layers[index + 1]
                cons_layer.rbm.model_distribution.bias_visible = np.copy(layer.rbm.model_distribution.bias_hidden)

    def backfit(self, data, batch_size=10):
        idx = utils.prepare_batches(len(data), batch_size)
        for b_idx in idx:
            batch = data[b_idx[0]:b_idx[1], :]

            # wake phase
            for index in range(len(self.dbn_layers)-1):
                layer = self.dbn_layers[index]
                batch,_ = layer.wake_pass(batch)

            # sampling in the top layer:
            top_layer = self.dbn_layers[-1]
            top_layer.greedy_train(batch)
            hid_prob, _ = top_layer.rbm.model_distribution.state_h(batch)
            batch, _ = top_layer.rbm.model_distribution.state_v(hid_prob)

            # sleep phase
            for index in reversed(range((len(self.dbn_layers)-1))):
                layer = self.dbn_layers[index]
                batch,_ = layer.sleep_pass(batch)

            # plotting (axis=0 --> operation over rows, axis=1 --> operation over columns)
            # prob_v_mean = np.reshape(np.mean(batch, axis=0), (28,28))
            # plotting.plot_rbm_2layer(v_minus_prob=prob_v_mean,fignum=1)
            # time.sleep(0.1)