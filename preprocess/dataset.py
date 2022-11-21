from typing import List
from torch.utils.data import Dataset

from preprocess.prepare import Crate
from tokenizers import Tokenizer
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

import numpy as np
import torch

class TokenizedDataset(Dataset):
    def __init__(self, crates: List[Crate], tokenizer: Tokenizer, max_length: int, num_categories: int):
        crate_strings = [crate.processed_string() for crate in crates]
        tokens = tokenizer.encode_batch(crate_strings)

        self.tokens = [token.ids[:max_length] for token in tokens]
        categories = np.zeros((len(crates), num_categories), dtype=np.float32)
        for i, crate in tqdm(enumerate(crates), total=len(crates), desc="Generating Multi-Hot Encodings"):
            categories[i, crate.category_indices] = 1
        self.categories = categories
        self.num_categories = num_categories

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.categories[idx]

    @staticmethod
    def word_bag_collate_fn(batch):
        tokens, categories = zip(*batch)
        cat_tokens = []
        offsets = []
        offset = 0
        for token in tokens:
            cat_tokens.extend(token)
            offsets.append(offset)
            offset += len(token)
        return np.array(cat_tokens), np.array(offsets), np.array(categories)

class BertDataset(Dataset):
    def __init__(self, crates: List[Crate], tokenizer: DistilBertTokenizerFast, max_length: int, num_categories: int):
        crate_strings = [crate.processed_string() for crate in crates]
        print("Encoding")
        self.tokens = tokenizer(crate_strings, max_length= max_length, padding='max_length', return_token_type_ids=True, return_attention_mask=True, truncation=True)
        print("Generating Multi-Hot Encodings")
        categories = np.zeros((len(crates), num_categories), dtype=np.float32)
        for i, crate in tqdm(enumerate(crates), total=len(crates), desc="Generating Multi-Hot Encodings"):
            categories[i, crate.category_indices] = 1
        self.categories = categories
        self.num_categories = num_categories

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
       return {
            "ids": self.tokens["input_ids"][idx],
            "mask": self.tokens["attention_mask"][idx],
            "token_type_ids": self.tokens["token_type_ids"][idx],
            "categories": self.categories[idx]
       }

    @staticmethod
    def bert_collate_fn(batch):
        ids = [item["ids"] for item in batch]
        mask = [item["mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        categories = [item["categories"] for item in batch]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "categories": np.array(categories)
        }

        





    