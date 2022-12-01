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
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_categories)


    def forward(self, text: torch.Tensor, text_offsets: torch.Tensor, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
        # TODO: deps is not used yet
        X = self.embedding(text, text_offsets)
        X = self.activation(X)
        X = self.dropout(X)
        X_rc = self.fc1(X)
        X_rc = self.activation(X)
        X = X + X_rc
        X = self.dropout(X)
        X = self.fc2(X)
        return X
