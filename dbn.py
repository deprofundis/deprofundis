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

    def __init__(self, lrate, momentum, wcost, layer_units, n_epochs=100000, k=1):
        self.lrate = lrate
        self.momentum = momentum
        self.wcost = wcost
        self.n_epochs = n_epochs

        assert(isinstance(layer_units, list) and len(layer_units) > 2)
        self.layer_units = layer_units

        assert(k > 0)
        self.k = k

        # create rbms
        self.n_layers = len(layer_units) - 1
        self.rbms = []
        for n_v, n_h in zip(layer_units[:-1], layer_units[1:]):
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
            
    def up_pass(self, ):

    def down_pass(self, ):

    def run_cdk_on_top_layers(self, v_plus):
        # get v_plus 
        top_rbm = self.rbms[-1]
        v_minus = top_rbm.learn_trial(v_plus) # v_minus after k-step CD
        return v_minus
            
        

    
    
