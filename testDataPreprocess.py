from datasetPreprocess import dataset
import unittest
class testPreprocess(unittest.TestCase):
    data = dataset("training.1600000.processed.noemoticon.csv")
    def testConstructor(self):
        self.assertEquals(data.trainText)

    def testpreprocess(self):
