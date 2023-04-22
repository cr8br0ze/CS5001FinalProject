import torch

class tokenize:
    def __init__(self, list, maxLen, tokenizer):
        self.textlist = list.tolist()
        self.maxLength = maxLen
        self.bertTokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embedding = self.encoding()

    def encoding(self):
        tokens = self.bertTokenizer.__call__(
            self.textlist,
            add_special_tokens=True,
            max_length=self.maxLength,
            padding="max_length",
            truncation=True,
            return_tensors = "pt").to(self.device)
        return tokens

    def encodeResult(self):
        inputId= self.embedding['input_ids'].to(self.device)
        tokenTypeId = self.embedding['token_type_ids'].to(self.device)
        attentionMask = self.embedding['attention_mask'].to(self.device)
        return inputId, tokenTypeId, attentionMask


