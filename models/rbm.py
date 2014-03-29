from sampler import Sampler
from optimizer import SGD, DynamicSGD
from distribution import Distribution

import numpy as np
import plotting


class RBM():
    """
    Class represents a restricted boltzmann machine
    """
    def __init__(self, model_distribution, sampler, optimizer):
        assert (isinstance(model_distribution, Distribution))
        assert (isinstance(sampler, Sampler))
        assert (isinstance(optimizer, SGD))

        self.model_distribution = model_distribution
        self.sampler = sampler
        self.optimizer = optimizer

    def train_batch(self, data):
        # sample data from model
        hidden_0_probs, hidden_0_states, \
            hidden_k_probs, hidden_k_states, \
            visible_k_probs, visible_k_states = self.sampler.sample(data)

        # copmute deltas
        d_weight_update, d_bias_hidden_update, \
            d_bias_visible_update = self.optimizer.optimize(data, hidden_0_states, hidden_0_probs, hidden_k_probs,
                                                            hidden_k_states, visible_k_probs, visible_k_states)

        # update model values
        self.model_distribution.weights += d_weight_update
        self.model_distribution.bias_hidden += d_bias_hidden_update
        self.model_distribution.bias_visible += d_bias_visible_update

        return d_weight_update, d_bias_hidden_update, d_bias_visible_update

    def reconstruct(self, data, steps=1):
        for i in range(steps):
            hidden_prob, hidden_state = self.model_distribution.state_h(data)
            rec_prob, rec_state = self.model_distribution.state_v(hidden_state)

        return rec_prob, rec_state

    def compute_reconstruction_error(self, data):
        """
        Computes the reconstruction error for a given dataset
        @param data: dataset the check the model
        @return: Pixel-wise calculated reconstruction error
        """
        hidden_prob, hidden_state = self.model_distribution.state_h(data)
        rec_prob, rec_state = self.model_distribution.state_v(hidden_state)

        if self.model_distribution.get_distribution_type() is Distribution.Type.DISCRETE:
            return (np.sum(np.abs(data - rec_state)) / float(data.size)) * 100
        else:
            raise NotImplementedError

    def compute_likelihood(self):
        pass


class CRBM(object):
    """
    Class represents a conditional restricted boltzmann machine.
    """
    def __init__(self, model_distribution, sampler, optimizer):
        assert (isinstance(model_distribution, Distribution))
        assert (isinstance(sampler, Sampler))
        assert (isinstance(optimizer, DynamicSGD))

        self.model_distribution = model_distribution
        self.sampler = sampler
        self.optimizer = optimizer

    def train_batch(self):
        pass