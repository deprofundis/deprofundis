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

    def __init__(self, lrate, momentum, wcost, layer_units, n_in_minibatch=100, n_epochs=100000, k=1):
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost
        self.n_in_minibatch = n_in_minibatch
        self.n_epochs = n_epochs

        assert(isinstance(layer_units, list) and len(layer_units) > 2)
        self.layer_units = layer_units

        assert(k > 0)
        self.k = k

        # create rbms
        self.n_layers = len(layer_units)
        self.rbms = []
        for n_v, n_h in zip(layer_units[:-1], layer_units[1:]):
            rbm = RbmNetwork(n_v, n_h, n_sampling_steps=k)
            self.rbms.append(rbm)

    def test_trial(self):
        # go all the way up
        # go all the way down
        # calculate reconstruction error
        pass

    def learn_trial(self):
        # greedy_layerwise_training()
        # up_down_training()
        pass

    def train_greedy(self, dataset):
        layer_input = dataset
        for rbm in self.rbms:
            rbm.train(layer_input)
            layer_input = rbm.propagate_fwd(layer_input)

         
    def train_backfiting(self, patterns):    

        for _ in range(self.n_epochs):
            vis_states = grab_minibatch(patterns, self.n_in_minibatch)        
             # obtain intial values from bottom rbm
            _, _, _, vis_states, vis_probs, _, _, _, _ = self.rbms[0].k_gibbs_steps(vis_states, k=1)
            # start backfitting
            _, top_state = self.up_pass(vis_probs, vis_states)
            _, v_k_state = self.run_cdk_on_top_layer(top_state)
            self.down_pass(v_k_state)
                    
    def up_pass(self, h_0_probs, h_0_state):
        """
        Performs an up pass on the data and performs mean-field updates of
        generative weights
        """
        batch_size = len(h_0_state)

        for rbm in self.rbms[:-1]: # skip last RBM, we'll do CD-k there
            # compute forward activation
            h_1_probs = rbm.propagate_fwd(h_0_state)
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
        batch_size = len(h_1_probs)

        for rbm in reversed(self.rbms[:-1]): # go from (top layer - 1) to 0
            # compute backward activation
            h_0_probs = rbm.propagate_back(h_1_state)
            h_0_state = rbm.samplestates(h_0_probs)
            # perform recognition weight update
            diff_h_1 = (h_1_state - h_1_probs)
            delta_w = np.dot(h_0_state.T, diff_h_1)
            delta_bias_vis = diff_h_1
            delta_bias_hid = (h_0_state - h_0_probs)

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
        v_k_act, v_k_state = top_rbm.k_gibbs_steps(self, v_plus) # v_minus after k-step CD
        return v_k_act, v_k_state
