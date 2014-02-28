from ipdb import set_trace as pause
from random import choice
from utils.utils import grab_minibatch
from rbm import *
from base import Network
import numpy as np

# TODO
#
# DBN with L layers
#
# Greedy layer-wise training
#   learn the weights for each layer l
#   pass the hidden activations as inputs to the next level
# Up-down
#   propagate the input all the way forward to level L
#   k-steps of CD in the top two layers
#   propagate back and perform positive/negative learning for each layer
#   (calculate gradient and update weights)
#
#

class DeepBeliefRBM(object):
    def __init__(self, rbm, lrate, momentum, wcost):

        # assert that it is an rbm
        assert (isinstance(rbm, Network))

        self.rbm = rbm
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost

        # recognition weights and biases (used for inference and updated during wake phase)
        self.rec_weights = None
        self.rec_bias_hidden = None
        self.rec_bias_visible = None

        self.d_rec_weights = np.zeros(shape = self.rbm.w.shape)
        self.d_rec_bias_hidden = np.zeros(shape = self.rbm.b.shape)
        self.d_rec_bias_visible = np.zeros(shape = self.rbm.a.shape)

        # generative weights and biases (reverse order since it is propagate top down)
        # (used for sampling and updated during sleep phase)
        self.gen_weights = None
        self.gen_bias_hidden = None
        self.gen_bias_visible = None

        self.d_gen_weights = np.zeros(shape = self.rbm.w.shape)
        self.d_gen_bias_hidden = np.zeros(shape = self.rbm.b.shape)
        self.d_gen_bias_visible = np.zeros(shape = self.rbm.a.shape)

    # pass all the training data in train patterns
    def train_greedy(self, train_patterns, valid_patterns=None, test_patterns=None,
              n_train_epochs=1000, n_in_minibatch=10,
              should_print=None, should_plot=None):
        # Make sure that there is enough training data
        assert(len(train_patterns) >= n_in_minibatch)
	
	pause()
	
        # Train the RBM
        self.rbm.train(train_patterns, valid_patterns, test_patterns,
                                         n_train_epochs, n_in_minibatch, should_print, should_plot)
	
        # Split trained weights and biases
        # Dimension of rec_weights is (#visible x #hidden)
        self.rec_weights = np.copy(self.rbm.w)
        # Already transposing weights since they are used in the top-down pass
        # Dimension of gen_weights is (#hidden x #visible)
        self.gen_weights = np.copy(self.rbm.w.T)
        self.rec_bias_hidden = self.gen_bias_hidden = np.copy(self.b)
        self.gen_bias_visible = np.copy(self.a)

    def up_pass(self, batch, mean_field = True):
	pause()

	# computes the fan in into hidden nodes, fan_in_hidden.shape = batch_size x num_hidden
        fan_in_hidden = np.dot(batch, self.rec_weights)
        fan_in_hidden = np.apply_along_axis(lambda x : x + self.rec_bias_hidden.T, 0, fan_in_hidden)
        # compute the activation of each hidden node, hidden_act.shape = batch_size x hidden_act
        hidden_act = self.rbm.act_fn(fan_in_hidden)
        # comupte the state of the hidden node, hidden_state.shape = batch_size x num_hidden
        hidden_state = hidden_act > np.random.uniform(size=hidden_act.shape)
        
        # compute reconstruction fan-in
        fan_in_visible = np.dot(hidden_state, self.gen_weights)
        fan_in_visible = np.apply_along_axis(lambda x : x + self.rec_bias_visible.T, 0, fan_in_visible)
        # compute visible activation
        visible_act = self.rbm.act_fn(fan_in_visible)

        # calculate weight deltas, delta_w.shape = num_visble x num_hidden
        delta_w = np.dot((batch - visible_act).T, hidden_state)
        delta_w = (self.lrate / len(batch)) * delta_w - self.gen_weights * self.wcost + self.momentum * self.d_gen_weights
        # weight update and store old deltas
        self.gen_weights += delta_w
        self.d_gen_weights = delta_w

        # caluclate bias delta, delta_bias_visible.shape = num_visible x 1
        delta_bias_visible = (batch - visible_act).mean(axis=0).T
        delta_bias_visible = self.lrate * delta_bias_visible - self.wcost * self.gen_bias_visible + \
                            self.momentum * self.d_gen_bias_visible
        self.gen_bias_visible += delta_bias_visible
        self.d_gen_weights = np.copy(delta_bias_visible)

        # delta_bias_hidden.shape = num_hidden x 1
        delta_bias_hidden = (hidden_state - hidden_act).mean(acis=0).T
        delta_bias_hidden = self.lrate * delta_bias_hidden - self.wcost * self.gen_bias_hidden + \
                            self.momentum * self.d_gen_bias_hidden
        self.gen_bias_hidden += delta_bias_hidden
        self.d_gen_bias_hidden = np.copy(delta_bias_hidden)

        return hidden_act if mean_field else hidden_state

    def down_pass(self, batch, mean_field = False):
	pause()

        # computes the fan in into the visible node
        fan_in_visible = np.dot(batch, self.gen_weights)
        fan_in_visible = np.apply_along_axis(lambda x : x + self.gen_bias_visible.T, 0, fan_in_visible)
        # compute the activation of the visible nodes
        visible_act = self.rbm.act_fn(fan_in_visible)
        # comute the states of the visible nodes
        visible_state = visible_act > np.random.uniform(size=visible_act.shape)

        # compute hidden fan in for reconstruction
        fan_in_hidden = np.dot(visible_state, self.rec_weights)
        fan_in_hidden = np.apply_along_axis(lambda x : self.gen_bias_hidden.T, 0, fan_in_hidden)
        # compute hidden activation
        hidden_act = self.rbm.act_fn(fan_in_hidden)
        hidden_state = hidden_act > np.random.uniform(size = hidden_act.shape)

        # calculate weight delta
        delta_w = np.dot((batch - hidden_act).T, visible_state)
        delta_w = (self.lrate / len(batch)) * delta_w - self.rec_weights * self.wcost + self.momentum * self.d_rec_weights
        # weight update
        self.rec_weights += delta_w
        self.d_rec_weights = delta_w

        # calculate hidden bias delta
        delta_bias_hidden = (hidden_state - hidden_act).mean(acis=0).T
        delta_bias_hidden = self.lrate * delta_bias_hidden - self.wcost * self.rec_bias_hidden + \
                            self.momentum * self.d_rec_bias_hidden
        self.rec_bias_hidden += delta_bias_hidden
        self.d_rec_bias_hidden = np.copy(delta_bias_hidden)

        # caluclate bias delta, delta_bias_visible.shape = num_visible x 1
        delta_bias_visible = (batch - visible_act).mean(axis=0).T
        delta_bias_visible = self.lrate * delta_bias_visible - self.wcost * self.rec_bias_visible + \
                            self.momentum * self.d_rec_bias_visible
        self.rec_bias_visible += delta_bias_visible
        self.d_rec_weights = np.copy(delta_bias_visible)

        return visible_act if mean_field else visible_state

class DBN(Network):
    def __init__(self, lrate, momentum, wcost, layer_units, n_in_minibatch=10, k=1, v_shape_bottom=None):
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost
        self.n_in_minibatch = n_in_minibatch
        self.v_shape_bottom = v_shape_bottom

        assert(isinstance(layer_units, list) and len(layer_units) > 2)
        self.layer_units = layer_units
        self.n_l = len(self.layer_units)

        assert(k > 0)
        self.k = k

        # create rbms
        self.n_layers = len(layer_units)
        self.n_rbms = self.n_layers-1
        self.layers = []
        for n_v, n_h in zip(layer_units[:-1], layer_units[1:]):
            rbm = RbmNetwork(n_v, n_h, n_sampling_steps=k)
            rbmLayer = DeepBeliefRBM(rbm, 0.001, 0, 0)
            self.layers.append(rbmLayer)

        if self.v_shape_bottom: self.layers[0].v_shape = self.v_shape_bottom

    def test_trial(self, v_input):
        # go all the way up
        # go all the way down
        # calculate reconstruction error
        raise NotImplementedError()

    def learn_trial(self):
        # greedy_layerwise_training()
        # up_down_training()
        raise NotImplementedError()

    def train_greedy(self, train_patterns, n_train_epochs, valid_patterns=None, should_plot=None, should_print=None):
       	
	pause()
	if not n_train_epochs:
            print 'Skipping greedy training'
            return
        layer_input = train_patterns
        for counter, rbmLayer in enumerate(self.layers):
            print 'Training greedy %i x %i net (%i of %i RBMs)' % \
                (rbmLayer.rbm.n_v, rbmLayer.rbm.n_h, counter, len(self.layers))

            rbmLayer.train_greedy(layer_input,
                      n_train_epochs=n_train_epochs,
                      should_print=should_print,
                      should_plot=should_plot)
            _, layer_input = rbmLayer.rbm.propagate_fwd(layer_input)

        if valid_patterns is not None:
            rand_ix = choice(range(len(valid_patterns)))
            self.plot_layers(valid_patterns[rand_ix, :], ttl='Greedy after %i epochs' % n_train_epochs)

    def train_backfitting(self, train_patterns, n_train_epochs, \
                    valid_patterns=None, should_print=None, should_plot=None):
        for epochnum in range(n_train_epochs):
            vis_states = grab_minibatch(train_patterns, self.n_in_minibatch)
            for batch in vis_states:
                # make up pass by ignoring the last layer
                for rbmLayer in self.layers[:-1]:
                    batch = rbmLayer.up_pass(batch)
                # sample at the top
                _, _, _, batch, _, _, _, _, _ = self.layers[-1].rbm.k_gibbs_steps(batch)
                # make down pass
                for rbmLayer in reversed(self.layers[:-1]):
                    batch = rbmLayer.down_pass(batch)

            if should_print and should_print(epochnum):
                print 'Training backfitting (%i of %i epochs)' % (epochnum, n_train_epochs)
            if should_plot and should_plot(epochnum) and valid_patterns is not None:
                rand_ix = choice(range(len(valid_patterns)))
                self.plot_layers(valid_patterns[rand_ix, :], ttl='Backfitting after %i epochs' % epochnum)


    def propagate_fwd_all(self, acts):
        for rbm in self.layers:
            _, acts = rbm.propagate_fwd(acts)
        return acts

    def propagate_back_all(self, h_1_probs):
        for rbm in reversed(self.layers):
            h_1_state = rbm.samplestates(h_1_probs)
            _, h_1_probs = rbm.propagate_back(h_1_state)
        return h_1_probs

    def plot_layers(self, v_plus_bottom, ttl=None):
        h_plus_top = self.propagate_fwd_all(v_plus_bottom)
        v_minus_bottom = self.propagate_back_all(h_plus_top)
        v_shape = self.v_shape_bottom
        bottom_rbm = self.layers[0]
    	plot_rbm_2layer(v_plus_bottom.reshape(v_shape),
                        None, None, None,
                        None, v_minus_bottom.reshape(v_shape), v_minus_bottom.reshape(v_shape),
                        None, None, None,
                        fignum=bottom_rbm.fignum_layers,
                        ttl=ttl)


if __name__ == "__main__":
    np.random.seed()

    lrate = 0.01
    wcost = 0.0002
    nhidden = 200
    n_trainpatterns = 500
    n_validpatterns = 1000
    n_train_greedy_epochs = 300
    n_train_backfitting_epochs = 10000
    n_in_minibatch = 5
    momentum = 0.9
    #n_temperatures = 1
    def n_temperatures(net):
        """
        Returns integer; 1 for single tempering
        """
        #M = net.trial_num / 500
        return 1# None #M if M > 1 else None

    #n_sampling_steps = 1
    def n_sampling_steps(net):
        """
        Returns integer; number of sampling steps used in CD-k
        """
        return 1

    plot = True
    plot_every_n = 100
    should_plot = lambda n: not n % plot_every_n # e.g. 0, 100, 200, 300, 400, ...
    print_every_n = 100
    should_print = lambda n: not n % print_every_n # e.g. 0, 100, 200, 300, 400, ...

    train_pset, valid_pset, test_pset = create_mnist_patternsets(n_trainpatterns=n_trainpatterns, n_validpatterns=n_validpatterns)
    n_trainpatterns, n_validpatterns, n_testpatterns = train_pset.n, valid_pset.n, test_pset.n

    net_t = Stopwatch()
    net = DBN(lrate, momentum, wcost, [784, 500, 200], v_shape_bottom=(28,28))
    print 'Init net in %.1fs' % net_t.finish(milli=False), net
    print 'train:', train_pset, '\tvalid:', valid_pset, '\ttest:', test_pset

    train_patterns = np.array(train_pset.patterns).reshape((n_trainpatterns,-1))
    valid_patterns = np.array(valid_pset.patterns).reshape((n_validpatterns,-1))
    test_patterns = np.array(test_pset.patterns).reshape((n_testpatterns, -1))

    net.train_greedy(train_patterns,
                     n_train_epochs=n_train_greedy_epochs,
                     valid_patterns=valid_patterns,
                     should_print=lambda n: not n % 100,
                     should_plot=lambda n: not n % 100)
    net.train_backfitting(train_patterns,
                          n_train_epochs=n_train_backfitting_epochs,
                          valid_patterns=valid_patterns,
                          should_plot=lambda n: not n % 100,
                          should_print=lambda n: not n % 100)
    
    pause()
