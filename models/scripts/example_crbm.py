from models.sampler import DynamicBlockGibbsSampler
from models.distribution import DynamicBernoulli
from models.optimizer import DynamicSGD
from utils.utils import prepare_frames
from scipy import io as matio
from data.gwtaylor.path import *
import ipdb
import numpy as np

SIZE_BATCH = 10
EPOCHS = 100
SIZE_HIDDEN = 50
SIZE_VISIBLE = 150

# CRBM Constants
M_LAG_VISIBLE = 2
N_LAG_HIDDEN = 2
SIZE_LAG = max(M_LAG_VISIBLE, N_LAG_HIDDEN)+1

# load and prepare dataset from .mat
mat = matio.loadmat(MOCAP_SAMPLE)
dataset = mat['batchdatabinary']
# generate batches
batch_idx_list = prepare_frames(len(dataset), SIZE_LAG, SIZE_BATCH)

# load distribution
bernoulli = DynamicBernoulli(SIZE_VISIBLE, SIZE_HIDDEN, m_lag_visible=M_LAG_VISIBLE, n_lag_hidden=N_LAG_HIDDEN)
gibbs_sampler = DynamicBlockGibbsSampler(bernoulli, sampling_steps=1)
sgd = DynamicSGD(bernoulli)

for epoch in range(EPOCHS):
    error = 0.0
    for chunk_idx_list in batch_idx_list:
        
        # get batch data set
        data = np.zeros(shape=(SIZE_BATCH, SIZE_VISIBLE, SIZE_LAG))
        for idx, (start, end) in enumerate(chunk_idx_list):
            data[idx, :, :] = dataset[start:end, :].T
        
        hidden_0_probs, hidden_0_states, \
            hidden_k_probs, hidden_k_states, \
            visible_k_probs, visible_k_states = gibbs_sampler.sample(data[:, :, 0], data[:, :, 1:])

        # compute deltas
        d_weight_update, d_bias_hidden_update, \
            d_bias_visible_update, d_vis_vis, d_vis_hid = sgd.optimize(data[:, :, 0], hidden_0_states, hidden_0_probs, hidden_k_probs,
                                                            hidden_k_states, visible_k_probs, visible_k_states, data[:, :, 1:])

        # update model values
        bernoulli.weights += d_weight_update
        bernoulli.bias_hidden += d_bias_hidden_update
        bernoulli.bias_visible += d_bias_visible_update
        bernoulli.vis_vis_weights += d_vis_vis
        bernoulli.vis_hid_weights += d_vis_hid
        
        # compute reconstruction error
        _, _, \
            _, _, \
            _, visible_k_states = gibbs_sampler.sample(data[:, :, 0], data[:, :, 1:])
            
        error += np.mean(np.abs(visible_k_states - data[:, :, 0]))
        
    error = 1./len(batch_idx_list) * error;
    print error
        