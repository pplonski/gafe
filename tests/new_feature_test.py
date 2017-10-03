'''
NewFeature tests.
'''
import os
import unittest
import numpy as np

from gafe.new_feature import NewFeature

class NewFeatureTest(unittest.TestCase):

    def test_init(self):
        nf = NewFeature()
        self.assertTrue(nf._operator_index is None)
        self.assertEqual(len(nf._columns), 0)
        self.assertTrue(nf.key() is None)


    def test_random_init(self):
        nf = NewFeature()
        nf.random_init(10)
        self.assertTrue(nf.key() is not None)

    def test_apply(self):
        nf = NewFeature()
        nf._operator_index = 0 # add
        nf._columns = [0, 1]
        array = np.ones((10, 3))
        r = nf.apply(array)
        self.assertEqual(np.sum(r), 20)

        nf._operator_index = 2 # substract
        r = nf.apply(array)
        self.assertEqual(np.sum(r), 0)

    def test_str(self):
        nf = NewFeature()
        nf._operator_index = 0 # add
        nf._columns = [0, 1, 2]
        self.assertEqual(nf.name(), 'col_0+col_1+col_2')

        nf = NewFeature()
        nf._operator_index = 1 # multiply
        nf._columns = [0, 1]
        self.assertEqual(nf.name(), 'col_0*col_1')
