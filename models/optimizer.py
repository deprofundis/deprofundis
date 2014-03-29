from distribution import Distribution

import numpy as np

class SGD(object):
    """
    Implements a version of stochastic gradient descent with the option to also use the stochastic gradient ascent,
    if the normal max. log-likelihood procedure is applied (in case of the negative log-likelihood, it would be
    the other way around).
    """

    def __init__(self, model_distribution, learning_rate=0.01, weight_decay=0.0002, momentum=0.9, is_ascent=True):
        assert (isinstance(model_distribution, Distribution))

        self.model_distribution = model_distribution

        # otherwise the model won't learn
        assert (learning_rate > 0)

        self.learning_rate = learning_rate
        self.weigth_decay = weight_decay
        self.momentum = momentum

        # a premultiplication factor to retieve the correct update rule
        self.ascent_factor = 1 if is_ascent else -1

        # initialize matrices to store deltas past values
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
        # assert correct dimensions
        assert (all(dim == visible_0_states.ndim for dim in (visible_k_probs.ndim, visible_k_states.ndim)))
        assert (all(dim == hidden_0_states.ndim for dim in (hidden_0_probs.ndim, hidden_k_probs.ndim,
                                                            hidden_k_states.ndim)))
        # assert inputs for correct shapes dimensions
        assert (self.model_distribution.size_visible in visible_0_states.shape)
        assert (self.model_distribution.size_visible in visible_k_states.shape)
        assert (self.model_distribution.size_visible in visible_k_probs.shape)
        assert (self.model_distribution.size_hidden in hidden_0_states.shape)
        assert (self.model_distribution.size_hidden in hidden_0_probs.shape)
        assert (self.model_distribution.size_hidden in hidden_k_probs.shape)
        assert (self.model_distribution.size_hidden in hidden_k_states.shape)
        # assert correct batch size
        if visible_k_probs.ndim > 1:
            assert (all(batch == len(visible_0_states) for batch in [len(hidden_0_states), len(hidden_0_probs),
                                                                     len(hidden_k_probs), len(hidden_k_states),
                                                                     len(visible_k_probs), len(visible_k_states)]))

        # calculate update for weights
        d_weight_update = (np.dot(visible_0_states.T, hidden_0_probs) -
                           np.dot(visible_k_states.T, hidden_k_probs)) / float(len(visible_0_states))
        d_weight_update = self.ascent_factor * self.learning_rate * d_weight_update \
                          - self.weigth_decay * self.model_distribution.weights \
                          + self.momentum * self.d_weights

        # calculate update for hidden biases
        d_bias_hidden_update = np.mean(hidden_0_probs - hidden_k_probs, axis=0)
        d_bias_hidden_update = self.ascent_factor * self.learning_rate * d_bias_hidden_update \
                               - self.weigth_decay * self.model_distribution.bias_hidden \
                               + self.momentum * self.d_bias_hidden

        # Calculate update for visible bias
        d_bias_visible_update = np.mean(visible_0_states - visible_k_states, axis=0)
        d_bias_visible_update = self.ascent_factor * self.learning_rate * d_bias_visible_update \
                                - self.weigth_decay * self.model_distribution.bias_visible \
                                + self.momentum * self.d_bias_visible
        # store values
        self.d_weights = np.copy(d_weight_update)
        self.d_bias_hidden = np.copy(d_bias_hidden_update)
        self.d_bias_visible = np.copy(d_bias_visible_update)

        return d_weight_update, d_bias_hidden_update, d_bias_visible_update
        
class DynamicSGD(object):
    pass