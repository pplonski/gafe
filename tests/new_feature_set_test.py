'''
NewFeatureSet tests.
'''
import os
import unittest
import tempfile
import numpy as np

from gafe.new_feature import NewFeature
from gafe.new_feature_set import NewFeatureSet

class NewFeatureSetTest(unittest.TestCase):

    def test_init(self):
        nfs = NewFeatureSet()

    def test_random_init(self):
        nfs = NewFeatureSet()
        self.assertTrue(nfs.key() is None)
        nfs.random_init(10, 5)
        self.assertTrue(nfs.key() is not None)

    def test_name(self):
        nfs = NewFeatureSet()
        nfs.random_init(10, 5)
        self.assertTrue(nfs.name() is not None)
        names = nfs.name()
        for nf in nfs._features:
            self.assertTrue(nf.name() in names)


    def test_apply(self):
        nfs = NewFeatureSet()
        for op in [0, 2]: # add, substract
            nf = NewFeature()
            nf._operator_index = op
            nf._columns = [0, 1]
            nfs._features += [nf]
            nfs._keys += [nf.key()]

        nf._columns = [0, 1]
        array = np.ones((10, 3))
        r = nfs.apply(array)
        self.assertEqual(r.shape[0], array.shape[0])
        self.assertEqual(r.shape[1], 2)
        self.assertEqual(np.sum(r), 20)


    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile() as temp:
            nfs = NewFeatureSet()
            nfs.random_init(10, 5)
            nfs.save(temp.name)
            nfs2 = NewFeatureSet()
            nfs2.load(temp.name)
            self.assertEqual(nfs.key(), nfs2.key())
