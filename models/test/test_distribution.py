from models.distribution import *
from utils import activation as act_fcn

import unittest as unittest
import numpy as np

VISIBLE_UNITS = 10
HIDDEN_UNITS = 5

SHAPE = (VISIBLE_UNITS, HIDDEN_UNITS)

class DistributionTest(unittest.TestCase):
    """
    Test class for the distribution base class
    """

    def test_correct_weight_matrix_dimension(self):
        """
        test if the matrix dimension have been implemented correctly
        @return: test result
        """
        dist = Distribution(VISIBLE_UNITS, HIDDEN_UNITS)
        self.assertEqual(dist.weights.shape,
                         (VISIBLE_UNITS, HIDDEN_UNITS),
                         "Weight matriy has wrong shape. Should be (size visible layer) x (size hidden layer)."
                         " Actual shape: " + str(dist.weights.shape))

    def test_correct_bias_hidden_dimension(self):
        dist = Distribution(VISIBLE_UNITS, HIDDEN_UNITS)
        self.assertEqual (dist.bias_hidden.shape, (HIDDEN_UNITS,),
                          "Hidden bias vector has wring shape. Should be (size hidden layer) x 1. "
                          "Actual shape: " + str(dist.bias_hidden.shape))

    def test_correct_dimension_bias_visible(self):
        dist = Distribution(VISIBLE_UNITS, HIDDEN_UNITS)
        self.assertEqual(dist.bias_visible.shape, (VISIBLE_UNITS,),
                         "Visible bias vector has wrong shape. Should be (size visible layer) x 1. "
                         "Actual shape: " + str(dist.bias_visible.shape))

    def test_error_wrong_weight_matrix_dim(self):
        weights = np.zeros(shape=(15,13))
        self.assertRaises(AssertionError, Distribution, VISIBLE_UNITS, HIDDEN_UNITS, weights)

    def test_error_wrong_hidden_bias_dim(self):
        bias_hidden = np.ones(shape=(12))
        self.assertRaises(AssertionError, Distribution, VISIBLE_UNITS, HIDDEN_UNITS, weights=None, bias_hidden=bias_hidden)

    def test_assert_wrong_visible_bias_dim(self):
        bias_visible = np.ones(shape=15)
        self.assertRaises(AssertionError, Distribution, VISIBLE_UNITS, HIDDEN_UNITS, weights=None, bias_hidden=None, bias_visible=bias_visible)


class BernoulliTest(unittest.TestCase):
    """
    Test class for the Bernoulli distribution implementation
    """
    weights = np.ones(shape=SHAPE)
    bias_hidden = np.ones(shape=HIDDEN_UNITS)
    bias_visible = np.ones(shape=VISIBLE_UNITS)
    vis_inp = np.ones(shape=VISIBLE_UNITS)
    hid_inp = np.ones(shape=HIDDEN_UNITS)

    def test_correct_score(self):
        """
        Tests whether the correct value for the energy function is returned. All properties (
        """
        dist = Bernoulli(VISIBLE_UNITS, HIDDEN_UNITS, weights=self.weights, bias_hidden=self.bias_hidden, bias_visible=self.bias_visible)
        target = np.sum(self.weights) + np.sum(self.bias_hidden) + np.sum(self.bias_visible)
        actual = np.abs(dist.score_energy(self.vis_inp, self.hid_inp))
        self.assertEqual(target, actual, "Didn't match. Target: " + str(target) + ", Actual: " + str(actual))

    def test_correct_conditional_prob_v(self):
        """
        Test whether the conditional probability p(v=1|h) is correctly computed. This function uses the fact that
        exp(0) = 1 and sigm(1) = 0.5
        """
        weights = self.weights * (1 / float(HIDDEN_UNITS))
        bias_visible = self.bias_visible * -1
        dist = Bernoulli(VISIBLE_UNITS, HIDDEN_UNITS, weights, self.bias_hidden, bias_visible)
        target = 0.5 * VISIBLE_UNITS
        actual = dist.conditional_prob_v(self.hid_inp)
        self.assertEqual(target, np.sum(actual), "Target: " + str(target) + ", Actual: " + str(actual))

    def test_correct_conditional_prob_h(self):
        """
        Test wheather the condtional probability p(h=1|v) is correctly computed. This function uses the fact that
        exp(0) = 1 and sigm(1) = 0.5
        """
        weights = self.weights * (1 / float(VISIBLE_UNITS))
        bias_hidden = self.bias_hidden * -1
        dist = Bernoulli(VISIBLE_UNITS, HIDDEN_UNITS, weights, bias_hidden, self.bias_visible)
        target = 0.5 * HIDDEN_UNITS
        actual = dist.conditional_prob_h(self.vis_inp)
        self.assertEqual(target, np.sum(actual), "Target: " + str(target) + ", Actual: " + str(actual))

    def test_correct_state_v(self):
        """
        Tests weather the state of the units is returned properly
        """
        pass

    def test_correct_state_h(self):
        """
        Tests weather the state of the units is returned properly
        """
        pass

    def test_correct_distribution_type(self):
        dist = Bernoulli(VISIBLE_UNITS, HIDDEN_UNITS)
        self.assertEqual(Distribution.Type.DISCRETE, dist.get_distribution_type())

if __name__ == '__main__':
    unittest.main()