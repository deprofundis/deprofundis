import inspect, os

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'csv')

MNIST_SAMPLE = os.path.join(BASE_PATH, 'sample_train.csv')
MNIST_SAMPLE_BIN = os.path.join(BASE_PATH, 'sample_train_binary.csv')

MNIST_TRAIN = os.path.join(BASE_PATH, 'train.csv')
MNIST_TEST = os.path.join(BASE_PATH, 'test.csv')

MNIST_TRAIN_BINARY = os.path.join(BASE_PATH, 'train_binary.csv')
MNIST_TEST_BINARY = os.path.join(BASE_PATH, 'test_binary.csv')