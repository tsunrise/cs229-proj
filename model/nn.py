import torch
from torch import nn

class NNModel(nn.Module):
    def __init__(self, num_words: int, num_categories: int):
        """
        Neural Net Model using bag of words
        - Input: A vector of word counts X, where X[i] is the number of times the i-th word appears in the message.
        - Output: logits score for a category.
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_words, 512, mode='sum')
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, num_categories)



    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        X = self.embedding(indices, offsets)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc1(X)
        return X
