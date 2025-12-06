import pandas as pd
import torch
from torch.utils.data import Dataset
import json

class IMDBDataset(Dataset):
    """Load dataset from multiple JSON files OR from pre-loaded lists"""

    def __init__(self, vocab, data_paths=None, tokenized_fpath=None, 
                 preloaded_reviews=None, preloaded_sentiments=None):
        """
        @param vocab (Vocabulary)
        @param data_paths (dict): (Legacy) Mapping sentiment to file path.
        @param preloaded_reviews (list): List of tokenized reviews (strings).
        @param preloaded_sentiments (list): List of sentiment labels.
        """
        self.vocab = vocab
        self.pad_idx = vocab["<pad>"]
        
        sentiments_type = ['negative', 'neutral', 'positive']
        self.sentiment2id = {sentiment: i for i, sentiment in enumerate(sentiments_type)}

        self.reviews_list = []
        self.sentiments_list = []

        # Ưu tiên sử dụng dữ liệu đã load và xử lý từ bên ngoài
        if preloaded_reviews is not None and preloaded_sentiments is not None:
            self.tokenized_reviews = preloaded_reviews
            self.sentiments_list = preloaded_sentiments
        else:
            # Fallback: Load từ file
            for sentiment, fpath in data_paths.items():
                with open(fpath, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                    self.reviews_list.extend(reviews)
                    self.sentiments_list.extend([sentiment] * len(reviews))

            if tokenized_fpath:
                self.tokenized_reviews = torch.load(tokenized_fpath)
            else:
                self.tokenized_reviews = self.vocab.tokenize_corpus(self.reviews_list)

        # Convert to tensors
        self.tensor_data = self.vocab.corpus_to_tensor(self.tokenized_reviews, is_tokenized=True)
        self.tensor_label = torch.tensor([self.sentiment2id[sentiment] for sentiment in self.sentiments_list],
                                         dtype=torch.long)
        
    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx], self.tensor_label[idx]

    def collate_fn(self, examples):
        # Sort batch by sequence length (descending) for pack_padded_sequence
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        reviews = [e[0] for e in examples]
        reviews = torch.nn.utils.rnn.pad_sequence(reviews, 
                                                  batch_first=False, 
                                                  padding_value=self.pad_idx)
        reviews_lengths = torch.tensor([len(e[0]) for e in examples])
        sentiments = torch.tensor([e[1] for e in examples])

        return {"reviews": (reviews, reviews_lengths), "sentiments": sentiments}