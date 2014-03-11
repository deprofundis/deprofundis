from utils import activation as act_fn
import numpy as np

class Distribution(object):
    """
    Interface for a basic probability distribution used in Energy based models.
    """

    def __init__(self, size_visible, size_hidden, weights=None, bias_hidden=None, bias_visible=None):
        self.size_visible = size_visible
        self.size_hidden = size_hidden

        # Initialize hidden bias
        if bias_hidden is None:
            self.bias_hidden = np.zeros(size_hidden)
        else:
            assert (bias_hidden.shape == (size_hidden,))
            self.bias_hidden = np.copy(bias_hidden)

        # Initial hidden bias
        if bias_visible is None:
            self.bias_visible = np.zeros(size_visible)
        else:
            assert (bias_visible.shape == (size_visible,))
            self.bias_visible = np.copy(bias_visible)

        # Initialize weight matrix
        if weights is None:
            self.weights = np.random.uniform(0, 1, size=(size_visible, size_hidden))
        else:
            assert (weights.shape == (size_visible, size_hidden))
            self.weights = np.copy(weights)

    def score_energy(self, visible, hidden):
        """
        Returns the value of the energy function
        @param visible: state of the visible units
        @param hidden: state of the hidden units
        """
        raise NotImplementedError

    def conditional_prob_v(self, hidden):
        """
        Returns the conditional probability for the visible units
        @param hidden: state of the hidden units
        """
        raise NotImplementedError

    def conditional_prob_h(self, visible):
        """
        Returns the conditional probability for the hidden units
        @param visible: state of the visible units
        """
        raise NotImplementedError

    def state_v(self, hidden):
        """
        Returns the state of the visible units
        @param hidden: state of the hidden units
        """
        raise NotImplementedError

    def state_h(self, visible):
        """
        Returns state of hidden units
        @param visible: state of the visible units
        """
        raise NotImplementedError

    def free_energy(self, visible, hidden):
        """
        Returns the free energy

        """
        raise NotImplementedError

    def get_distribution_type(self):
        """
        @return the type of the distribution
        """
        raise NotImplementedError

    class Type(object):
        DISCRETE = 1
        CONTINUOUS = 2

class Bernoulli(Distribution):
    """
    Implements a Bernoulli hidden visible connection
    """
    def __init__(self, size_visible, size_hidden, weights=None, bias_hidden=None, bias_visible=None):
        Distribution.__init__(self, size_visible, size_hidden, weights, bias_hidden, bias_visible)

    def score_energy(self, visible, hidden):
        # assert the correct shapes of the inputs
        assert (visible.ndim == hidden.ndim)
        assert (self.size_visible in visible.shape)
        assert (self.size_hidden in hidden.shape)
        # assert that batch size are equal, (but only if it is larger 1 otherwise the method fails
        if visible.ndim > 1:
            assert (len(hidden) == len(visible))

        # returns a vector (size batch) x 1
        return(-1) * (np.dot(visible, self.bias_visible)
                      + np.dot(hidden, self.bias_hidden)
                      + np.dot(np.dot(visible, self.weights), hidden.T))


    def conditional_prob_v(self, hidden):
        # assert correct shape (size of batch) x (size of hidden layer)
        assert (self.size_hidden in hidden.shape)
        # returns a matrix (size of batch) x (size of visible layer)
        return act_fn.sigmoid(self.bias_visible + np.dot(hidden, self.weights.T))

    def conditional_prob_h(self, visible):
        # assert correct shape (size of batch) x (size of visible layer)
        assert (self.size_visible in visible.shape)
        # returns a matrix (size of batch) x (size of hidden layer)
        return act_fn.sigmoid(self.bias_hidden + np.dot(visible, self.weights))

    def state_v(self, hidden):
        # assert correct shape (size of batch) x (size of hidden layer)
        assert (self.size_hidden in hidden.shape)
        # compute the conditional probability of visible units
        conditional_prob_v = self.conditional_prob_v(hidden)
        state_v = conditional_prob_v > np.random.uniform(0,1,size=(len(hidden), self.size_visible))
        # returns a matrix (size of batch) x (size of visible layer)
        return conditional_prob_v, state_v.astype(int)

    def state_h(self, visible):
        # assert correct shape (size of batch) x (size of visible layer)
        assert (self.size_visible in visible.shape)
        # compute the conditional probability of hidden units
        conditional_prob_h = self.conditional_prob_h(visible)
        state_h = conditional_prob_h > np.random.uniform(0,1,size=(len(visible), self.size_hidden))
        # returns a matrix (size of batch) x (size of hidden layer)
        return conditional_prob_h, state_h.astype(int)

    def free_energy(self, visible, hidden):
        # assert the correct shapes of the inputs
        assert (visible.ndim == hidden.ndim)
        assert (self.size_visible in visible.shape)
        assert (self.size_hidden in hidden.shape)
        # assert that batch size are equal, (but only if it is larger 1 otherwise the method fails
        if visible.ndim > 1:
            assert (len(hidden) == len(visible))
        raise NotImplementedError("Implement before using")

    def get_distribution_type(self):
        return Distribution.Type.DISCRETE

