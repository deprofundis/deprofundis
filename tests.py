from ipdb import set_trace as pause
from nose.tools import set_trace as pause, ok_, eq_
import numpy as np
from numpy.testing import assert_array_equal as arr_eq_
import unittest

from rbm import RbmNetwork, create_random_patternset


class BaseTestCase(unittest.TestCase):
    pass


class RbmTests(BaseTestCase):
    def setUp(self):
        self.net = RbmNetwork(3, 2, 0.01, 0.001)
        self.pset = create_random_patternset(shape=(1,3), npatterns=4)


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.pset = create_random_patternset(shape=(1,5), npatterns=10)

    def test_getmulti(self):
        patterns_single = [self.pset.get(p) for p in range(5)]
        patterns_multi = self.pset.getmulti(range(5))
        for pattern_single, pattern_multi in zip(patterns_single, patterns_multi):
            arr_eq_(pattern_single, pattern_multi)
            

if __name__ == '__main__':
    unittest.main()
