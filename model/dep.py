import torch
from torch import nn
import torch.nn.functional as F
class FeatNet(nn.Module):
    def __init__(self, num_dep_words: int, hidden_dim: int, dropout_p) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_dep_words, hidden_dim, mode='sum')
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
        X = self.embedding(deps, deps_offsets)
        X = F.leaky_relu(X)
        X = self.dropout(X)
        X = self.fc1(X)
        return X