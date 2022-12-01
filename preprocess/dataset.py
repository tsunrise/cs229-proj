from typing import List, Tuple
from torch.utils.data import Dataset

from preprocess.prepare import Crate
from tokenizers import Tokenizer
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

import numpy as np
import torch
from preprocess.tokenize import PADDING_TOKEN, PADDING_TOKEN_ID

def flatten_to_get_offsets(lst: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - lst: 2d list of ints of potentially different lengths
    Returns: 
    - tensor of flattened list
    - tensor of offsets
    """
    flattened = []
    offsets = []
    offset = 0
    for l in lst:
        flattened.extend(l)
        offsets.append(offset)
        offset += len(l)
    return torch.tensor(flattened), torch.tensor(offsets)

class CrateDataset(Dataset):
    def __init__(self, crates: List[Crate], text_tokenizer: Tokenizer, deps_tokenizer: Tokenizer, num_categories: int, seq_length: int = 128):
        crate_strings = [crate.processed_string() for crate in crates]
        text_tokenizer.enable_padding(pad_id=PADDING_TOKEN_ID, pad_token=PADDING_TOKEN, length=seq_length)
        text_tokens = text_tokenizer.encode_batch(crate_strings)
        # truncate to seq_length
        self.text_tokens = [x.ids[:seq_length] for x in text_tokens]

        categories = np.zeros((len(crates), num_categories), dtype=int)
        for i, crate in tqdm(enumerate(crates), total=len(crates), desc="Generating Multi-Hot Encodings"):
            categories[i, crate.category_indices] = 1
        self.categories = categories
        self.num_categories = num_categories

        crate_deps = [" ".join(crate.dependency) for crate in crates]
        deps_tokens = deps_tokenizer.encode_batch(crate_deps)
        self.deps_tokens = [x.ids for x in deps_tokens]

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        return {
            "text": self.text_tokens[idx],
            "deps": self.deps_tokens[idx],
            "categories": self.categories[idx]
        }

    @staticmethod
    def word_bag_collate_fn(batch):
        text_tokens = [x["text"] for x in batch]
        deps_tokens = [x["deps"] for x in batch]
        categories = [x["categories"] for x in batch]

        text_tokens, text_offsets = flatten_to_get_offsets(text_tokens)
        deps_tokens, deps_offsets = flatten_to_get_offsets(deps_tokens)
        categories = np.array(categories)
        categories = torch.tensor(categories, dtype=torch.float32)

        return {
            "text": text_tokens,
            "text_offsets": text_offsets,
            "deps": deps_tokens,
            "deps_offsets": deps_offsets,
            "categories": categories
        }

    @staticmethod
    def seq_collate_fn(batch):
        text_tokens = [x["text"] for x in batch]
        deps_tokens = [x["deps"] for x in batch]
        categories = [x["categories"] for x in batch]

        text_tokens = torch.tensor(text_tokens)
        deps_tokens, deps_offsets = flatten_to_get_offsets(deps_tokens)
        categories = torch.tensor(categories).float()

        return {
            "text": text_tokens,
            "deps": deps_tokens,
            "deps_offsets": deps_offsets,
            "categories": categories
        }

class BertDataset(Dataset):
    def __init__(self, crates: List[Crate], tokenizer: DistilBertTokenizerFast, deps_tokenizer: Tokenizer, max_length: int, num_categories: int):
        crate_strings = [crate.processed_string() for crate in crates]
        print("Encoding")
        self.tokens = tokenizer(crate_strings, max_length= max_length, padding='max_length', return_token_type_ids=True, return_attention_mask=True, truncation=True)
        print("Generating Multi-Hot Encodings")
        categories = np.zeros((len(crates), num_categories), dtype=np.float32)
        for i, crate in tqdm(enumerate(crates), total=len(crates), desc="Generating Multi-Hot Encodings"):
            categories[i, crate.category_indices] = 1
        self.categories = categories
        self.num_categories = num_categories

        crate_deps = [crate.dependency for crate in crates]
        deps_tokens = deps_tokenizer.encode_batch(crate_deps)
        self.deps_tokens = [x.ids for x in deps_tokens]

    def __len__(self):
        return len(self.tokens["input_ids"]) 

    def __getitem__(self, idx):
       return {
            "ids": self.tokens["input_ids"][idx],
            "mask": self.tokens["attention_mask"][idx],
            "token_type_ids": self.tokens["token_type_ids"][idx],
            "categories": self.categories[idx],
            "deps": self.deps_tokens[idx]
       }

    @staticmethod
    def bert_collate_fn(batch):
        ids = [item["ids"] for item in batch]
        mask = [item["mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        categories = [item["categories"] for item in batch]

        deps_tokens = [item["deps"] for item in batch]
        deps_tokens, deps_offsets = flatten_to_get_offsets(deps_tokens)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "categories": np.array(categories),
            "deps": deps_tokens,
            "deps_offsets": deps_offsets
        }

        





    