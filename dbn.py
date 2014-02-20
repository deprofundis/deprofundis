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
# For simplicity: constant number of units per layer
class DeepBeliefRBM(object):
    def __init__(self, rbm, lrate, momentum, wcost):

        # assert that it is an rbm
        assert (isinstance(rbm, Network))

        self.rbm = rbm
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost

        # recognition weights and biases

        self.rec_weights = None
        self.rec_bias_hidden = None
        self.rec_bias_visible = None

        self.d_rec_weights = np.zeros(shape = self.rbm.w.shape)
        self.d_rec_bias_hidden = np.zeros(shape = self.rbm.b.shape)
        self.d_rec_bias_visible = np.zeros(shape = self.rbm.a.shape)

        # generative weights and biases (reverse order since it is propagate top down
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
        self.rec_bias_visible = self.gen_bias_visible = np.copy(self.a)

    def up_pass(self, batch):
        # computes the fan in into hidden nodes, fan_in_hidden.shape = batch_size x num_hidden
        fan_in_hidden = np.dot(batch, self.rec_weights) + np.kron(np.ones((len(batch),1), self.rec_bias_hidden.T))
        # compute the activation of each hidden node, hidden_act.shape = batch_size x hidden_act
        hidden_act = self.rbm.act_fn(fan_in_hidden)
        # comupte the state of the hidden node, hidden_state.shape = batch_size x num_hidden
        hidden_state = hidden_act > np.random.uniform(size=hidden_act.shape)
        # compute reconstruction fan-in
        fan_in_visible = np.dot(hidden_state, self.gen_weights) + np.kron(np.ones((len(batch),1), self.gen_bias_visible.T))
        # compute visible activation
        visible_act = self.rbm.act_fn(fan_in_visible)

        # calculate weight deltas, delta_w.shape = num_visble x num_hidden
        delta_w = np.dot((batch - visible_act).T, hidden_state,)
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

    def down_pass(self, batch):
        fan

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
        self.rbms = []
        for n_v, n_h in zip(layer_units[:-1], layer_units[1:]):
            rbm = RbmNetwork(n_v, n_h, n_sampling_steps=k)
            self.rbms.append(rbm)
        if self.v_shape_bottom: self.rbms[0].v_shape = self.v_shape_bottom

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
        if not n_train_epochs:
            print 'Skipping greedy training'
            return
        layer_input = train_patterns
        nRBMs = len(self.rbms)
        for counter, rbm in enumerate(self.rbms):
            print 'Training greedy %i x %i net (%i of %i RBMs)' % \
                (rbm.n_v, rbm.n_h, counter, self.n_rbms)
            rbm.train(layer_input,
                      n_train_epochs=n_train_epochs,
                      should_print=should_print,
                      should_plot=should_plot)
            _, layer_input = rbm.propagate_fwd(layer_input)
        if valid_patterns is not None:
            rand_ix = choice(range(len(valid_patterns)))
            self.plot_layers(valid_patterns[rand_ix, :], ttl='Greedy after %i epochs' % n_train_epochs)

    def train_backfitting(self, train_patterns, n_train_epochs, \
                    valid_patterns=None, should_print=None, should_plot=None):
        for epochnum in range(n_train_epochs):

            if should_print and should_print(epochnum):
                print 'Training backfitting (%i of %i epochs)' % (epochnum, n_train_epochs)
            if should_plot and should_plot(epochnum) and valid_patterns is not None:
                rand_ix = choice(range(len(valid_patterns)))
                self.plot_layers(valid_patterns[rand_ix, :], ttl='Backfitting after %i epochs' % epochnum)
            vis_states = grab_minibatch(train_patterns, self.n_in_minibatch)        
             # obtain intial values from bottom rbm
            _, _, _, _, vis_probs, vis_states, _, _, _ = self.rbms[0].k_gibbs_steps(vis_states, k=1)
            # start backfitting
            _, top_state = self.up_pass(vis_probs, vis_states)
            v_k_probs, v_k_state = self.run_cdk_on_top_layers(top_state)
            self.down_pass(v_k_probs, v_k_state)
                    
    def up_pass(self, h_0_probs, h_0_state):
        """
        Performs an up pass on the data and performs mean-field updates of
        generative weights
        """
        batch_size = len(h_0_state)

        for rbm in self.rbms[:-1]: # skip last RBM, we'll do CD-k there
            # compute forward activation
            _, h_1_probs = rbm.propagate_fwd(h_0_state)
            h_1_state = rbm.samplestates(h_1_probs)

            # perform generative weight update
            diff_h0 = (h_0_state - h_0_probs)
            delta_w = np.dot(diff_h0.T, h_1_state)
            delta_bias_vis = diff_h0
            delta_bias_hid = (h_1_state - h_1_probs)

            delta_w = self.lrate / batch_size * delta_w
            delta_bias_vis = (self.lrate / batch_size * delta_bias_vis).mean(axis=0)
            delta_bias_hid = (self.lrate / batch_size * delta_bias_hid).mean(axis=0)

            print "Up ", delta_w

            #rbm.w = rbm.w + delta_w
            #rbm.a = rbm.a + delta_bias_vis
            #rbm.b = rbm.b + delta_bias_hid

            # feed activations into higher-level RBM
            h_0_probs = h_1_probs
            h_0_state = h_1_state

        return h_1_probs, h_1_state

    def down_pass(self, h_1_probs, h_1_state):
        """
        Performs a downward pass on the data and performs mean-field updates of
        the network's recognition weights
        """
        batch_size = len(h_1_state)

        for rbm in reversed(self.rbms[:-1]): # go from (top layer - 1) to 0
            # compute backward activation
            _, h_0_probs = rbm.propagate_back(h_1_state)
            h_0_state = rbm.samplestates(h_0_probs)
            # perform recognition weight update
            diff_h_1 = (h_1_state - h_1_probs)
            delta_w = np.dot(h_0_state.T, diff_h_1)
            delta_bias_vis = (h_0_state - h_0_probs)
            delta_bias_hid = diff_h_1

            delta_w = self.lrate / batch_size * delta_w
            delta_bias_vis = (self.lrate / batch_size * delta_bias_vis).mean(axis=0)
            delta_bias_hid = (self.lrate / batch_size * delta_bias_hid).mean(axis=0)

            print "Down ", delta_w
            
            #rbm.w = rbm.w + delta_w
            #rbm.a = rbm.a + delta_bias_vis
            #rbm.b = rbm.b + delta_bias_hid

            h_1_probs = h_0_probs
            h_1_state = h_0_state
        return h_0_probs, h_0_state

    def run_cdk_on_top_layers(self, v_plus):
        # get v_plus 
        top_rbm = self.rbms[-1]
        _, _, _, _, v_k_act, v_k_state, _, _, _ = top_rbm.k_gibbs_steps(v_plus) # v_minus after k-step CD
        return v_k_act, v_k_state

    def propagate_fwd_all(self, acts):
        for rbm in self.rbms:
            _, acts = rbm.propagate_fwd(acts)
        return acts

    def propagate_back_all(self, h_1_probs):
        for rbm in reversed(self.rbms):
            h_1_state = rbm.samplestates(h_1_probs)
            _, h_1_probs = rbm.propagate_back(h_1_state)
        return h_1_probs

    def plot_layers(self, v_plus_bottom, ttl=None):
        h_plus_top = self.propagate_fwd_all(v_plus_bottom)
        v_minus_bottom = self.propagate_back_all(h_plus_top)
        v_shape = self.v_shape_bottom
        bottom_rbm = self.rbms[0]
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