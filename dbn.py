from utils import grab_minibatch

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
        greedy_layerwise_training()
        up_down_training()

    def train_greedy(self, dataset):
        pass
         
    def train_backfiting(self, patterns):    
        for _ in range(n_epochs):
            vis_states = grab_minibatch(patterns, self.n_in_minibatch)        
             # obtain intial values from bottom rbm
            vis_act = self.rbms[0].we_dont_know_what_comes_here(vis_states)
            # start backfitting
            _, top_state = up_pass(vis_states, vis_act)
            _, v_k_state = run_cdk_on_top_layer(top_state)
            down_pass(v_k_state)
                    
    def up_pass(self, h_0_act, h_0_state):
        """
        Performs an up pass on the data and performs mean-field updates of
        generative weights
        """
        for rbm in self.rbms[:-1]: # skip last RBM, we'll do CD-k there
            # compute forward activation
            h_1_act = rbm.propagate_fwd(h_0_state)
            h_1_state = rbm.samplestates(h_1_act)
            # perform generative weight update
            delta_w = self.lrate * h_1_state * (h_0_state - h_0_act)
            rbm.update_weights_with_delta(delta_w)
            # feed activations into higher-level RBM
            h_0_act = h_1_act
            h_0_state = h_1_state

        return h_1_act, h_1_state


    def down_pass(self, h_1_act, h_1_state):
        """
        Performs a downward pass on the data and performs mean-field updates of
        the network's recognition weights
        """
        for rbm in reversed(self.rbms[:-1]): # go from (top layer - 1) to 0
            # compute backward activation
            h_0_act = rbm.propagate_back(h_1_state)
            h_0_state = rbm.samplestates(h_0_act)
            # perform recognition weight update
            delta_w = self.lrate * h_0_state * (h_1_state - h_1_act)
            rbm.update_weights_with_delta(delta_w)
            h_1_act = h_0_act
            h_1_state = h_0_state
        return h_0_act, h_0_state

    def run_cdk_on_top_layers(self, v_plus):
        # get v_plus 
        top_rbm = self.rbms[-1]
        v_k_act, v_k_state = top_rbm.k_gibbs_steps(self, v_plus) # v_minus after k-step CD
        return v_k_act, v_k_state
        
if __mame__ == "__main__":
    
    
