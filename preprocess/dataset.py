from typing import List
from torch.utils.data import Dataset

from preprocess.prepare import Crate
from preprocess.tokenize import MyTokenizer

import numpy as np
class BagOfWordsDataset(Dataset):
    def __init__(self, crates: List[Crate], tokenizer: MyTokenizer, max_length: int, num_categories: int):
        tokens = []
        for crate in crates:
            assert crate.processed
            tokens.append(tokenizer.encode_crate(crate)[:max_length])

        self.tokens = tokens
        categories = np.zeros((len(crates), num_categories), dtype=np.float32)
        for i, crate in enumerate(crates):
            categories[i, crate.category_indices] = 1
        self.categories = categories

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        return self.tokens[idx], self.categories[idx]

    @staticmethod
    def collate_fn(batch):
        tokens, categories = zip(*batch)
        cat_tokens = []
        offsets = []
        offset = 0
        for token in tokens:
            cat_tokens.extend(token)
            offsets.append(offset)
            offset += len(token)
        return np.array(cat_tokens), np.array(offsets), np.array(categories)
    