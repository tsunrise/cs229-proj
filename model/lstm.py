import torch
from torch import nn
from torch.nn import functional as F

from model.dep import DepNet

class LSTMModel(nn.Module):
    def __init__(self, num_words: int, num_dep_words: int, num_categories: int, hidden_size: int, dropout_p: float) -> None:
        super().__init__()
        # text is a sequence of word indices
        # deps is a word bag of dependency indices
        self.depnet = DepNet(num_dep_words, hidden_size, dropout_p=dropout_p)

        self.embedding = nn.Embedding(num_words, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(3 * hidden_size, num_categories)

    def forward(self, text: torch.Tensor, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
        # deps is a word bag of dependency indices
        X_dep = self.depnet(deps, deps_offsets)

        X = self.embedding(text)
        _, (X, _) = self.lstm1(X)
        X = self.dropout(X)
        X = torch.cat([X[0], X[1], X_dep], dim=1)
        X = self.fc(X)
        return X






