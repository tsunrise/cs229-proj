
from sklearn.model_selection import train_test_split

def train_dev_split(data_seq, train_ratio=0.8, seed=0):
    train, dev = train_test_split(data_seq, train_size=train_ratio, random_state=seed)
    return train, dev