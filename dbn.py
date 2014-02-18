from random import sample
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

class DBN(Network):
    def __init__(self, lrate, momentum, wcost, layer_units, n_in_minibatch=10, n_epochs=1000, k=1, v_shape_bottom=None):
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost
        self.n_in_minibatch = n_in_minibatch
        self.n_epochs = n_epochs
        self.v_shape_bottom = v_shape_bottom

        assert(isinstance(layer_units, list) and len(layer_units) > 2)
        self.layer_units = layer_units

        assert(k > 0)
        self.k = k

        # create rbms
        self.n_layers = len(layer_units)
        self.n_rbms = self.n_layers-1
        self.rbms = []
        for n_v, n_h in zip(layer_units[:-1], layer_units[1:]):
            rbm = RbmNetwork(n_v, n_h, n_sampling_steps=k)
            self.rbms.append(rbm)

    def test_trial(self, v_input):
        # go all the way up
        # go all the way down
        # calculate reconstruction error
        raise NotImplementedError()

    def learn_trial(self):
        # greedy_layerwise_training()
        # up_down_training()
        raise NotImplementedError()

    def train_greedy(self, dataset, n_train_epochs, should_plot=None, should_print=None):
        if not n_train_epochs:
            print 'Skipping greedy training'
            return
        layer_input = dataset
        for counter, rbm in enumerate(self.rbms):
            print 'Training greedy %i x %i net (%i of %i RBMs)' % \
                (rbm.n_v, rbm.n_h, counter, self.n_rbms)
            rbm.train(layer_input,
                      n_train_epochs=n_train_epochs,
                      should_print=should_print,
                      should_plot=should_plot)
            _, layer_input = rbm.propagate_fwd(layer_input)

    def train_backfitting(self, train_patterns, valid_patterns=None, should_print=None, should_plot=None):
        for epochnum in range(self.n_epochs):
            if should_print and should_print(epochnum):
                print 'Training backfitting (%i of %i epochs)' % (epochnum, self.n_epochs)
            if should_plot and should_plot(epochnum) and valid_patterns is not None:
                rand_ix = sample(range(len(valid_patterns)), 1)
                #self.rbms[0].plot_layers(valid_patterns[rand_ix, :])
            vis_states = grab_minibatch(train_patterns, self.n_in_minibatch)        
             # obtain intial values from bottom rbm
            _, _, _, vis_states, vis_probs, _, _, _, _ = self.rbms[0].k_gibbs_steps(vis_states, k=1)
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
            delta_bias_vis = self.lrate / batch_size * delta_bias_vis
            delta_bias_hid = self.lrate / batch_size * delta_bias_hid

            rbm.w = rbm.w + delta_w
            rbm.a = rbm.a + delta_bias_vis
            rbm.b = rbm.b + delta_bias_hid

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
            delta_bias_vis = self.lrate / batch_size * delta_bias_vis
            delta_bias_hid = self.lrate / batch_size * delta_bias_hid

            rbm.w = rbm.w + delta_w
            rbm.a = rbm.a + delta_bias_vis
            rbm.b = rbm.b + delta_bias_hid

            h_1_probs = h_0_probs
            h_1_state = h_0_state
        return h_0_probs, h_0_state

    def run_cdk_on_top_layers(self, v_plus):
        # get v_plus 
        top_rbm = self.rbms[-1]
        _, _, _, _, v_k_act, v_k_state, _, _, _ = top_rbm.k_gibbs_steps(v_plus) # v_minus after k-step CD
        return v_k_act, v_k_state


if __name__ == "__main__":
    np.random.seed()

    lrate = 0.01
    wcost = 0.0002
    nhidden = 200
    n_trainpatterns = 5000
    n_validpatterns = 1000
    n_train_epochs = 250
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
    net = DBN(lrate, momentum, wcost, [784, 500, 200], n_epochs=n_train_epochs, v_shape_bottom=(28,28))
    print 'Init net in %.1fs' % net_t.finish(milli=False), net
    print 'train:', train_pset, '\tvalid:', valid_pset, '\ttest:', test_pset

    train_patterns = np.array(train_pset.patterns).reshape((n_trainpatterns,-1))
    valid_patterns = np.array(valid_pset.patterns).reshape((n_validpatterns,-1))
    test_patterns = np.array(test_pset.patterns).reshape((n_testpatterns, -1))

    net.train_greedy(train_patterns,
                     n_train_epochs=500,
                     should_print=lambda n: not n % 100,
                     should_plot=lambda n: not n % 100)
    net.train_backfitting(train_patterns,
                          valid_patterns=valid_patterns,
                          should_plot=lambda n: not n % 100,
                          should_print=lambda n: not n % 100)
    
    pause()


