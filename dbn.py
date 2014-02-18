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

    def __init__(self, n_layers, n_units):
        self.up_epochs = 100000
        self.down_epochs = 100000

        self.rbms = []
        for l in n_layers:
            n_v = n_units # number of input units?
            n_h = n_units
            rbm = RbmNetwork(n_v, n_h)
            self.rbms.append(rbm)

    def test_trial(self):
        # go all the way up
        # go all the way down
        # calculate reconstruction error
        pass

    def learn_trial(self):
        greedy_layerwise_training()
        up_down_training()

    def greedy_layerwise_training(self, wholeds):
        layer_input = wholeds
        for rbm in self.rbms:
            rbm.train(layer_input)
            layer_input = rbm.propagatefwd(layer_input)
            
    def up_down_training(self, wholeds):
        up_activations = propagate_fwd_all(wholeds) #propagate all the forward to get v+ for top rbm
        v_minus_top = run_cdk_on_top_layers(v_plus)
        for rbm in self.rbms[-2::-1]: # iterate backwards ignoring top RBM
            # take up_activations for this rbm
            # take sample from higher-level rbm
            # calculate gradient and adjust weights

    def run_cdk_on_top_layers(self, v_plus):
        # get v_plus 
        top_rbm = self.rbms[-1]
        v_minus = top_rbm.learn_trial(v_plus) # v_minus after k-step CD
        return v_minus
            
        

    
    
