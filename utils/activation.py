import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input matrix
    @return: value of the sigmioid function at x
    """
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    """
    First derivative of the sigmoid activation function
    @param x: input matrix
    @return: value of the derivative function at x
    """
    s = sigmoid(x)
    return s * (1. - s)

def tanh(x):
    """
    Tangens hyperbolicus activation function
    @param x: input matrix
    @return: value of the tanh function at x
    """
    return np.tanh(x)

def d_tanh(x):
    """
    First derivative of the tangens hyperbolicus function
    @param x: input matrix
    @return: value of the derivative at x
    """
    return 1. - np.power(tanh(x), 2)
