from textEmbedding import tokenize
import numpy as np
import torch
from transformers import BertTokenizer
import unittest


class testTextEmbedding(unittest.TestCase):
    def setUp(self):
        list = np.array(["I love CS5001", "Zichong Meng"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.tokenized = tokenize(list, 5, self.tokenizer)

    def testEncoding(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tokens = self.tokenizer.__call__(["I love CS5001", "Zichong Meng"],
                           add_special_tokens=True,
                           max_length=5,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt").to(device)
        for i in tokens.keys():
            self.assertTrue(i in ["input_ids", "token_type_ids", "attention_mask"])
            self.assertEqual(tokens[i].shape, torch.Size([2, 5]))

    def test_encodeResult(self):
        inputId, tokenTypeId, attentionMask = self.tokenized.encodeResult()
        self.assertEqual(inputId.shape, torch.Size([2, 5]))
        self.assertEqual(tokenTypeId.shape, torch.Size([2, 5]))
        self.assertEqual(attentionMask.shape, torch.Size([2, 5]))

if __name__ == '__main__':
    unittest.main()