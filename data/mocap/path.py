import inspect,os

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

# sample mocap data proviede by Taylor (http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html)
# data descrption from (http://people.csail.mit.edu/ehsu/work/sig05stf/)
MOCAP_SAMPLE = os.path.join(BASE_PATH, 'sample.mat')
