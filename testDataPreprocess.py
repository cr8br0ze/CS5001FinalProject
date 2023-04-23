from datasetPreprocess import twitterData
import unittest


class testPreprocess(unittest.TestCase):
    def setUp(self):
        self.path = "training.1600000.processed.noemoticon.csv"
        self.data = twitterData(self.path)

    def testPreprocess(self):
        dataset = self.data.preprocess("training.1600000.processed.noemoticon.csv")
        self.assertEqual(dataset.shape, (1600000, 2))
        self.assertListEqual(list(dataset.columns), ["score", "text"])
        self.assertEqual(self.data.dataset.shape, (1600000, 2))
        self.assertListEqual(list(self.data.dataset.columns), ["score", "text"])

    def test_split(self):
        trainText, valText, testText, trainScore, valScore, testScore =\
            self.data.split(self.data.preprocess("training.1600000.processed.noemoticon.csv"))
        self.assertEqual(trainText.shape, (int(1600000*0.8*0.8),))
        self.assertEqual(valText.shape, (int(1600000 * 0.8 * 0.2),))
        self.assertEqual(testText.shape, (int(1600000 * 0.2),))
        self.assertEqual(trainScore.shape, (int(1600000 * 0.8 * 0.8),))
        self.assertEqual(valScore.shape, (int(1600000 * 0.8 * 0.2),))
        self.assertEqual(testScore.shape, (int(1600000 * 0.2),))
        self.assertEqual(self.data.trainText.shape, (int(1600000*0.8*0.8),))
        self.assertEqual(self.data.valText.shape, (int(1600000 * 0.8 * 0.2),))
        self.assertEqual(self.data.testText.shape, (int(1600000 * 0.2),))
        self.assertEqual(self.data.trainScore.shape, (int(1600000*0.8*0.8),))
        self.assertEqual(self.data.valScore.shape, (int(1600000 * 0.8 * 0.2),))
        self.assertEqual(self.data.testScore.shape, (int(1600000 * 0.2),))

    def testMaxLen(self):
        self.data.trainText = ["I Love CS5001", ""]
        self.assertEqual(self.data.maxLen(self.data.trainText), 3)


if __name__ == '__main__':
    unittest.main()
