from scipy import io as matio
from data.mocap.path import *

# Motion is a cell array containing 3 sequences of walking motion (120fps)
# skel is struct array which describes the person's joint hierarchy
def load_mocap_sample():
    mat = matio.loadmat(MOCAP_SAMPLE)
    return mat

lol = load_mocap_sample()
print ""