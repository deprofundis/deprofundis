from sampler import Sampler
from optimizer import Optimizer
from distribution import Distribution

import numpy as np

class RBM():
    def __init__(self, model_distribution, sampler, optimizer):
        assert (isinstance(model_distribution, Distribution))
        assert (isinstance(sampler, Sampler))
        assert (isinstance(optimizer, Optimizer))

        self.model_distribution = model_distribution
        self.sampler = sampler
        self.optimizer = optimizer

    def train_batch(self, data):
        hidden_0_probs, hidden_0_states, \
            hidden_k_probs, hidden_k_states, \
            visible_k_probs, visible_k_states = self.sampler.sample(data)

        d_weight_update, d_bias_hidden_update, \
            d_bias_visible_update = self.optimizer.optimize(data, hidden_0_states, hidden_0_probs, hidden_k_probs,
                                                            hidden_k_states, visible_k_probs, visible_k_states)

        return d_weight_update, d_bias_hidden_update, d_bias_visible_update

    def compute_reconstruction_error(self, data):
        """
        Computes the reconstruction error for a given dataset
        @param data: dataset the check the model
        @return: Pixel-wise calculated reconstruction error
        """
        if self.model_distribution.get_distribution_type() is Distribution.Type.DISCRETE:
            hidden_prob, hidden_state = self.model_distribution.state_h(data)
            rec_prob, rec_state = self.model_distribution.state_v(hidden_state)

            return (np.sum(np.abs(data - rec_state)) / float(data.size))
        else:
            raise NotImplementedError



    def compute_likelihood(self):
        pass