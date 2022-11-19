
from typing import List, Tuple, TypeVar
from sklearn.model_selection import train_test_split

T = TypeVar('T')

def train_dev_split(data_seq: List[T], train_ratio=0.8, seed=0) -> Tuple[List[T], List[T]]:
    train, dev = train_test_split(data_seq, train_size=train_ratio, random_state=seed)
    return train, dev