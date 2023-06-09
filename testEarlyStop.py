"""
    4/23/2023
    CS5001
    Zichong Meng
    This program file is for Final Project testing
"""

#imports
from earlyStop import earlyStop
import numpy as np
import unittest

class testEarlyStop(unittest.TestCase):

    # to create the object I want to test
    def setUp(self):
        self.Stopper = earlyStop(3, 0)

    #test constructor of the earlyStop class
    def testConstructor(self):
        self.assertEqual(self.Stopper.patience, 3)
        self.assertEqual(self.Stopper.minDelta, 0)
        self.assertFalse(self.Stopper.stop)
        self.assertEqual(self.Stopper.minVal, np.inf)
        self.assertEqual(self.Stopper.count, 0)

    #test stopOrNot method of the earlyStop class
    def testStopOrNot(self):
        self.Stopper.stopOrNot(0.3)
        self.assertEqual(self.Stopper.count, 0)
        self.assertEqual(self.Stopper.minVal, 0.3)

        self.Stopper.stopOrNot(0.2)
        self.assertEqual(self.Stopper.count, 0)
        self.assertEqual(self.Stopper.minVal, 0.2)

        self.Stopper.stopOrNot(0.3)
        self.assertEqual(self.Stopper.count, 1)
        self.assertEqual(self.Stopper.minVal, 0.2)

        self.Stopper.stopOrNot(0.3)
        self.assertEqual(self.Stopper.count, 2)
        self.assertEqual(self.Stopper.minVal, 0.2)

        self.Stopper.stopOrNot(0.5)
        self.assertEqual(self.Stopper.count, 3)
        self.assertEqual(self.Stopper.minVal, 0.2)
        self.assertTrue(self.Stopper.stop)

    #test stopp method of the earlyStop class
    def testStopp(self):
        self.assertFalse(self.Stopper.stopp())
        self.Stopper.stopOrNot(0.2)
        self.Stopper.stopOrNot(0.1)
        self.Stopper.stopOrNot(0.3)
        self.Stopper.stopOrNot(0.3)
        self.Stopper.stopOrNot(0.3)
        self.assertTrue(self.Stopper.stopp())

if __name__ == '__main__':
    unittest.main()
