
from typing import List, Tuple, TypeVar
from sklearn.model_selection import train_test_split

T = TypeVar('T')

def train_dev_test_split(data_seq: List[T]) -> Tuple[List[T], List[T], List[T]]:
    train, dev_test = train_test_split(data_seq, train_size=0.7, random_state=0)
    dev, test = train_test_split(dev_test, train_size=0.5, random_state=0)
    return train, dev, test