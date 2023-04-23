"""
    4/23/2023
    CS5001
    Zichong Meng
    This program file is for Final Project testing
"""

#imports
from prediction import prediction
import unittest

class testPrediction(unittest.TestCase):

    #set up the predictor object to test
    def setUp(self):
        self.predictor = prediction()

    #tes the predict method of the prediction class
    def testPredict(self):
        output = self.predictor.predict("I love CS5001")
        self.assertEqual(output, "positive")
        output2 = self.predictor.predict("I hate")
        self.assertEqual(output2, "negative")

    #test the history method of the prediction class
    def testHistory(self):
        self.predictor.predict("I love CS5001")
        self.predictor.predict("I hate")
        self.assertEqual(self.predictor.history(), [{"I love CS5001":"positive"}, {"I hate":"negative"}])

    #test the print object result
    def testPrint(self):
        self.predictor.predict("I love CS5001")
        print(self.predictor)
        print("should be same as:\nyour history is:\ninput: I love CS5001 , result: positive")