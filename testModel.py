from model import modelFinetuning
import unittest

class testModelFinetuning(unittest.TestCase):

    def setUp(self):
        self.modelFinetuning = modelFinetuning()

    def testConsturctor(self):
        self.assertEqual(self.modelFinetuning.epoch, 13)
        self.assertFalse(self.modelFinetuning.stopped)
        self.assertEqual(self.modelFinetuning.tunedPath, [])

if __name__ == '__main__':
    unittest.main()
