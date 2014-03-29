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
            self.weights = np.random.uniform(-4 * np.sqrt(6. / (size_hidden + size_visible)),
                                             4 * np.sqrt(6. / (size_hidden + size_visible)),
                                             size=(size_visible, size_hidden))
        else:
            assert (weights.shape == (size_visible, size_hidden))
            self.weights = np.copy(weights)

    def score_energy(self, visible, hidden):
        """
        Returns the value of the energy function
        @param visible: state of the visible units
        @param hidden: state of the hidden units
        """
        # assert the correct shapes of the inputs
        assert (visible.ndim == hidden.ndim)
        assert (self.size_visible in visible.shape)
        assert (self.size_hidden in hidden.shape)

    def conditional_prob_v(self, hidden):
        """
        Returns the conditional probability for the visible units
        @param hidden: state of the hidden units
        """
        # assert correct shape (size of batch) x (size of hidden layer)
        assert (self.size_hidden in hidden.shape)

    def conditional_prob_h(self, visible):
        """
        Returns the conditional probability for the hidden units
        @param visible: state of the visible units
        """
        # assert correct shape (size of batch) x (size of visible layer)
        assert (self.size_visible in visible.shape)

    def state_v(self, hidden):
        """
        Returns the state of the visible units
        @param hidden: state of the hidden units
        """
        # assert correct shape (size of batch) x (size of hidden layer)
        assert (self.size_hidden in hidden.shape)

    def state_h(self, visible):
        """
        Returns state of hidden units
        @param visible: state of the visible units
        """
        assert (self.size_visible in visible.shape)

    def free_energy(self, visible, hidden):
        """
        Returns the free energy

        """
        # assert the correct shapes of the inputs
        assert (visible.ndim == hidden.ndim)
        assert (self.size_visible in visible.shape)
        assert (self.size_hidden in hidden.shape)

        # assert that batch size are equal, (but only if it is larger 1 otherwise the method fails
        if visible.ndim > 1:
            assert (len(hidden) == len(visible))

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
        super(Bernoulli, self).score_energy(visible, hidden)
        # assert that batch size are equal, (but only if it is larger 1 otherwise the method fails
        if visible.ndim > 1:
            assert (len(hidden) == len(visible))

        # returns a vector (size batch) x 1
        return(-1) * (np.dot(visible, self.bias_visible)
                      + np.dot(hidden, self.bias_hidden)
                      + np.dot(np.dot(visible, self.weights), hidden.T))


    def conditional_prob_v(self, hidden):
        super(Bernoulli, self).conditional_prob_v(hidden)
        # returns a matrix (size of batch) x (size of visible layer)
        fan = self.bias_visible + np.dot(hidden, self.weights.T)
        act = act_fn.sigmoid(fan)
        return act

    def conditional_prob_h(self, visible):
        super(Bernoulli, self).conditional_prob_h(visible)
        # returns a matrix (size of batch) x (size of hidden layer)
        fan = self.bias_hidden + np.dot(visible, self.weights)
        act = act_fn.sigmoid(fan)
        return act

    def state_v(self, hidden):
        super(Bernoulli, self).state_v(hidden)
        # compute the conditional probability of visible units
        conditional_prob_v = self.conditional_prob_v(hidden)
        state_v = conditional_prob_v > np.random.uniform(0,1,size=(len(hidden), self.size_visible))
        # returns a matrix (size of batch) x (size of visible layer)
        return conditional_prob_v, state_v.astype(int)

    def state_h(self, visible):
        super(Bernoulli, self).state_h(visible)
        # compute the conditional probability of hidden units
        conditional_prob_h = self.conditional_prob_h(visible)
        state_h = conditional_prob_h > np.random.uniform(0,1,size=(len(visible), self.size_hidden))
        # returns a matrix (size of batch) x (size of hidden layer)
        return conditional_prob_h, state_h.astype(int)

    def free_energy(self, visible, hidden):
        super(Bernoulli, self).free_energy(visible,hidden)
        raise NotImplementedError("Implement before using")

    def get_distribution_type(self):
        return Distribution.Type.DISCRETE
        
class DynamicBernoulli(Distribution):
    """
    Implements a distribution with dynamic biases
    """
    def __init__(self, size_visible, size_hidden, weights=None, vis_vis_weights=None, 
                 vis_hid_weights=None, bias_hidden=None, bias_visible=None,
                 m_lag_visible=1, n_lag_hidden=1):
        Distribution.__init__(self, size_visible, size_hidden, weights, bias_hidden, bias_visible)
        
        # dynamic biases
        self.m_lag_visible = m_lag_visible # m
        self.n_lag_hidden = n_lag_hidden # n
        
        # initialize weights
        if vis_vis_weights is None: # A
            self.vis_vis_weights = np.random.uniform(-4 * np.sqrt(6. / (size_visible + size_visible)),
                                             4 * np.sqrt(6. / (size_visible + size_visible)),
                                             size=(size_visible, size_visible, m_lag_visible)) # TODO initialization
        else:
            assert (vis_vis_weights.shape == (size_visible, size_visible, m_lag_visible))
            self.vis_vis_weights = np.copy(vis_vis_weights)
        
        if vis_hid_weights is None: # B
            self.vis_hid_weights = np.random.uniform(-4 * np.sqrt(6. / (size_hidden + size_visible)),
                                             4 * np.sqrt(6. / (size_hidden + size_visible)),
                                             size=(size_visible, size_hidden, n_lag_hidden)) # TODO initialization
        else:
            assert (vis_hid_weights.shape == (size_visible, size_hidden, n_lag_hidden))
            self.vis_hid_weights = np.copy(vis_hid_weights)
            
            
    def score_energy(self, visible, hidden):
        """
        @param visible: visible input (includes time-lagged values)
        @param hidden: hidden input (NO time-lagged values)
        """
        lag = max(self.n_lag_hidden, self.m_lag_visible)
        assert(visible.shape == (lag, self.size_visible))
        
        
    def conditional_prob_v(self, hidden, visible_lagged):
        """
        @param hidden: hidden input
        @param visible_lagged: lagged visible input (m input vectors)
        """
        assert(hidden.shape[-1] == (self.size_hidden))
        assert(visible_lagged.shape[1:] == (self.size_visible, self.m_lag_visible))
        assert(len(visible_lagged) == len(hidden))
        
        # calculate fan in to visible units
        fan_in = self.bias_visible
        for lag in range(self.m_lag_visible):
            fan_in += np.dot(visible_lagged[:,:,lag], self.vis_vis_weights[:,:,lag])

        fan_in += np.dot(self.weights, hidden.T)

        act = act_fn.sigmoid(fan_in)
        # returns a matrix of size (1xself.size_visible)
        return act

    def conditional_prob_h(self, visible, visible_lagged):
        """
        @param visible: visible input
        @param visible_lagged: lagged visible input(n input vectors)
        """
        assert(visible.shape[-1] == (self.size_visible))
        assert(visible_lagged.shape[1:] == (self.size_visible, self.n_lag_hidden))
        assert(len(visible_lagged) == len(visible))
        
        # calculate fan in to hidden units
        fan_in = self.bias_hidden
        for lag in range(self.n_lag_hidden):
            fan_in += np.dot(visible_lagged[:,:,lag], self.vis_hid_weights[:,:,lag])
        
        fan_in += np.dot(visible, self.weights)
        
        act = act_fn.sigmoid(fan_in)
        # returns a matrix of size (1xself.size_hidden)
        return act

    def state_v(self, hidden, visible_lagged):
        """
        @param hidden: hidden input
        @param visible_lagged: lagged visible input (m input vectors)
        """
        assert(hidden.shape[-1] == (self.size_hidden))
        assert(visible_lagged.shape[1:] == (self.size_visible, self.m_lag_visible))
        assert(len(visible_lagged) == len(hidden))
        
        # compute conditional probabilities of visible units
        conditional_prob_v = self.conditional_prob_v(hidden, visible_lagged)
        # sample states
        state_v = conditional_prob_v > np.random.uniform(0,1,size=(conditional_prob_v.shape))
        # returns a matrix (size of batch) x (size of visible layer)
        return conditional_prob_v, state_v.astype(int)
        
    def state_h(self, visible, visible_lagged):
        """
        @param visible: visible input
        @param visible_lagged: lagged visible input(n input vectors)
        """
        assert(visible.shape[-1] == (self.size_visible))
        assert(visible_lagged.shape[1:] == (self.size_visible, self.n_lag_hidden))
        assert(len(visible_lagged) == len(visible))
        
        # compute conditional probabilities of hidden units
        conditional_prob_h = self.conditional_prob_h(visible, visible_lagged)
        # sample states
        state_h = conditional_prob_h > np.random.uniform(0,1,size=(conditional_prob_h.shape))
        # returns a matrix (size of batch=1) x (size of hidden layer) # TODO include minibatches
        return conditional_prob_h, state_h.astype(int)
        
    def free_energy(self, visible, hidden):
        raise NotImplementedError("Not implemented")
        

    def get_distribution_type(self):
        return Distribution.Type.DISCRETE
        
# TODO Implement variance as parameter into distribution
class GaussianBinary(Bernoulli):
    """
    Implementes a Gaussian to binary distribution
    """
    def __init__(self, size_visible, size_hidden, std = 1, weights=None, bias_hidden=None, bias_visible=None):
        Bernoulli.__init__(self, size_visible, size_hidden, weights, bias_hidden, bias_visible)

        self.std = std

    def score_energy(self, visible, hidden):
        """
        Computes the energy function for a given input of visible and hidden data. The variance is assumed to be set
        to 1. Complete energy function looks as:

        E(v,h) = 0.5 * (v - bias_vis).T * (v - b) - bias_hid.T * h - v.T * diag(1/sigma_1, ..., 1/sigma_(#vis)) * W * h

        @param visible: Visible units input.
        @param hidden: Hidden units input.
        @return: Returns the energy of a gaussian-binary Restricted Boltzmann Machine.
        """
        super(GaussianBinary, self).score_energy(visible, hidden)
        zero_mean = visible - self.bias_visible
        return 0.5 * np.dot(zero_mean, zero_mean) - np.dot(self.bias_hidden, hidden) - np.dot((visible * self.weights), hidden)

    def state_v(self, hidden):
        super(GaussianBinary, self).state_v(hidden)
        # retrieve conditional probability
        cond_prob = self.conditional_prob_v(hidden)
        # add gaussian noise which create a gaussian distributed random variable
        return cond_prob, cond_prob + np.random.standard_normal(size=self.size_visible)

    def conditional_prob_v(self, hidden):
        super(GaussianBinary, self).conditional_prob_v(hidden)
        return self.bias_visible + np.dot(hidden, self.weights.T)

    def get_distribution_type(self):
        return Distribution.Type.CONTINUOUS