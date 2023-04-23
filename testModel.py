from model import modelFinetuning
import torch
import unittest


class TestModelFinetuning(unittest.TestCase):

    def setUp(self):
        self.modelFinetuning = modelFinetuning()

    def testConsturctor(self):
        self.assertEqual(self.modelFinetuning.epoch, 13)
        self.assertFalse(self.modelFinetuning.stopped)
        self.assertEqual(self.modelFinetuning.tunedPath, [])

if __name__ == '__main__':
    unittest.main()
