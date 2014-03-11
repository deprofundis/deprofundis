from models.optimizer import *

import unittest as unittest

class SGDTest(unittest.TestCase):
    """
    Test class for Stochastic gradient descent class
    """
    def test_learning_rate_smaller_zeroi(self):
        self.assertRaises(AssertionError, SGD, )