from typing import List
from torch.utils.data import Dataset

from preprocess.prepare import Crate
from preprocess.tokenize import MyTokenizer
from tqdm import tqdm

import numpy as np
class BagOfWordsDataset(Dataset):
    def __init__(self, crates: List[Crate], tokenizer: MyTokenizer, max_length: int, num_categories: int):
        for crate in crates:
            assert crate.processed
        tokens = tokenizer.encode_crates(crates, max_length)

        self.tokens = tokens
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
    