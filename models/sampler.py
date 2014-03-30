from distribution import Distribution

import numpy as np

class Sampler():
    """
    Interface for a sampling method
    """
    def __init__(self, model_distribution):
        assert (isinstance(model_distribution, Distribution))

        self.model_distribution = model_distribution

    def sample(self, initial):
        raise NotImplementedError


class BlockGibbsSampler(Sampler):
    """
    Implements a Block Gibbs Sampler.
    """
    def __init__(self, model_distribution, sampling_steps=1):
        Sampler.__init__(self, model_distribution)
        self.sampling_steps = sampling_steps

    def sample(self, visible_0_state):
        # assert that the initial data sample has the correct shape
        assert (self.model_distribution.size_visible in visible_0_state.shape)
        # calculate the initial state for the hidden unit
        hidden_0_prob, hidden_0_state = self.model_distribution.state_h(visible_0_state)
        # copy the value since a numpy array is a mutable object, there it will get changed during the sampling process
        # which we don't want
        visible_k_state = np.copy(visible_0_state)
        hidden_k_state = np.copy(hidden_0_state)
        # start block gibbs sampling procedure
        for step in range(self.sampling_steps):
            visible_k_prob, visible_k_state = self.model_distribution.state_v(hidden_k_state)
            hidden_k_prob, hidden_k_state = self.model_distribution.state_h(visible_k_state)

        return hidden_0_prob, hidden_0_state,\
               hidden_k_prob, hidden_k_state, \
               visible_k_prob, visible_k_state
               
class DynamicBlockGibbsSampler(Sampler):
    """
    Implements a Dynamic Block Gibbs Sampler (DBGS)
    """
    def __init__(self, model_distribution, sampling_steps=1):
        Sampler.__init__(self, model_distribution)
        self.sampling_steps = sampling_steps
        
    def sample(self, visible_0_state, visible_lagged):
        # assert that the initial data sample has the correct shape
        assert (self.model_distribution.size_visible in visible_0_state.shape)
        assert (len(visible_0_state) == len(visible_lagged))
        
        # calculate the initial state for the hidden unit
        hidden_0_prob, hidden_0_state = self.model_distribution.state_h(visible_0_state, visible_lagged)
        # copy the value since a numpy array is a mutable object, there it will get changed during the sampling process
        # which we don't want
        visible_k_state = np.copy(visible_0_state)
        hidden_k_state = np.copy(hidden_0_state)
        # start block gibbs sampling procedure
        for step in range(self.sampling_steps):
            visible_k_prob, visible_k_state = self.model_distribution.state_v(hidden_k_state, visible_lagged)
            hidden_k_prob, hidden_k_state = self.model_distribution.state_h(visible_k_state, visible_lagged)

        return hidden_0_prob, hidden_0_state,\
               hidden_k_prob, hidden_k_state, \
               visible_k_prob, visible_k_state
        