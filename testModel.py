"""
    4/23/2023
    CS5001
    Zichong Meng
    This program file is for Final Project testing
"""
#the model.py file only need to test few things because it is mainly using other classes which already tested
#and ultilzing pytorch library/framework to do the model finetuning(method/functions already tested by pytorch)
#and seeing the finetuning result is also a better way to know if the code works

#imports
from model import modelFinetuning
import unittest

class testModelFinetuning(unittest.TestCase):

    #set up the modelFinetuning object I want to test
    def setUp(self):
        self.modelFinetuning = modelFinetuning()

    #test the consturctor of the modelFinetuning class
    def testConsturctor(self):
        self.assertEqual(self.modelFinetuning.epoch, 13)
        self.assertFalse(self.modelFinetuning.stopped)
        self.assertEqual(self.modelFinetuning.tunedPath, [])

if __name__ == '__main__':
    unittest.main()
