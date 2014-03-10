from distribution import Distribution

import numpy as np

class Optimizer():
    """
    Interface for an omptimization method such as stochastic gradient descent
    """
    def __init__(self, model_distribution):
        assert (isinstance(model_distribution, Distribution))

        self.model_distribution = model_distribution

    def optimize(self, visible_0_states, hidden_0_states, hidden_0_probs, hidden_k_probs, hidden_k_states,
                 visible_k_probs, visible_k_states):
        raise NotImplementedError

class SGD(Optimizer):
    """
    Implements a version of stochastic gradient descent with the option to also use the stochastic gradient ascent,
    if the normal max. log-likelihood procedure is applied (in case of the negative log-likelihood, it would be
    the other way around).
    """
    def __init__(self, model_distribution, learning_rate=0.01, weight_decay=0.0002, momentum=0.9, is_ascent=True):
        Optimizer.__init__(self, model_distribution)

        # otherwise the model won't learn
        assert (learning_rate > 0)

        self.learning_rate = learning_rate
        self.weigth_decay = weight_decay
        self.momentum = momentum

        # a premultiplication factor to retieve the correct update rule
        self.ascent_factor = 1 if is_ascent else -1

        # initialize matrices to store deltas past values
        if momentum is not 0:
            self.d_weights = np.zeros(shape=self.model_distribution.weights.shape)
            self.d_bias_visible = np.zeros(shape=self.model_distribution.bias_visible.shape)
            self.d_bias_hidden = np.zeros(shape=self.model_distribution.bias_hidden.shape)

    def optimize(self, visible_0_states, hidden_0_states, hidden_0_probs, hidden_k_probs, hidden_k_states,
                 visible_k_probs, visible_k_states):
        """
        Optimizies the given paramters of the probability distribution through stochastic gradient descent given sampled
        data. It will update automatically the values of the model distribution

        @param visible_0_states: states of the visible units that were fed into the syste
        @param hidden_0_states: states of the hidden units after the initial sampling pass
        @param hidden_0_probs: probabilites (e.g. mean field activation) of the hidden units after a inital sampling pass
        @param hidden_k_probs: probabilities (e.g. mean field activation) of the hidden units after k sampling steps
        @param hidden_k_states: states of the hidden units after k sampling steps
        @param visible_k_probs: probabilites (e.g. mean field activation) of visible units after k sampling steps
        @param visible_k_states: states of the visible units after k sampling steps
        @return: delta values for parameter update
        """
        # get batch size
        batch_size = len(visible_0_states)
        # assert inputs for correct dimensions
        assert (visible_0_states.shape == (batch_size, self.model_distribution.size_visible))
        assert (visible_k_states.shape == (batch_size, self.model_distribution.size_visible))
        assert (visible_k_probs.shape == (batch_size, self.model_distribution.size_visible))
        assert (hidden_0_states.shape == (batch_size, self.model_distribution.size_hidden))
        assert (hidden_0_probs.shape == (batch_size, self.model_distribution.size_hidden))
        assert (hidden_k_probs.shape == (batch_size, self.model_distribution.size_hidden))
        assert (hidden_k_states.shape == (batch_size, self.model_distribution.size_hidden))

        # calculate update for weights
        d_weight_update = (np.dot(visible_0_states.T, hidden_0_probs) -
                           np.dot(visible_k_probs.T, hidden_k_probs)) / float(batch_size)
        d_weight_update = self.ascent_factor * self.learning_rate * d_weight_update \
                          - self.weigth_decay * self.model_distribution.weights \
                          + self.momentum * self.d_weights
        d_weight_update += self.d_weights

        # calculate update for hidden biases
        d_bias_hidden_update = np.mean(hidden_0_probs - hidden_k_probs,axis=0)
        d_bias_hidden_update = self.ascent_factor * self.learning_rate * d_bias_hidden_update \
                               - self.weigth_decay * self.model_distribution.bias_hidden \
                               + self.momentum * self.d_bias_hidden

        # Calculate update for visible bias
        d_bias_visible_update = np.mean(visible_0_states - visible_k_probs,axis=0)
        d_bias_visible_update = self.ascent_factor * self.learning_rate * d_bias_visible_update \
                               - self.weigth_decay * self.model_distribution.bias_visible \
                               + self.momentum * self.d_bias_visible
        # store values
        self.d_weights = np.copy(d_weight_update)
        self.d_bias_hidden = np.copy(d_bias_hidden_update)
        self.d_bias_visible = np.copy(d_bias_visible_update)

        # update model values
        self.model_distribution.weights += d_weight_update
        self.model_distribution.bias_hidden += d_bias_hidden_update
        self.model_distribution.bias_visible += d_bias_visible_update

        return d_weight_update, d_bias_hidden_update, d_bias_visible_update