from prediction import prediction
import unittest

class testPrediction(unittest.TestCase):
    def setUp(self):
        self.predictor = prediction()

    def testPredict(self):
        output = self.predictor.predict("I love CS5001")
        self.assertEqual(output, "positive")
        output2 = self.predictor.predict("I hate")
        self.assertEqual(output2, "negative")

    def test_history(self):
        self.predictor.predict("I love CS5001")
        self.predictor.predict("I hate")
        self.assertEqual(self.predictor.history(), [{"I love CS5001":"positive"}, {"I hate":"negative"}])
