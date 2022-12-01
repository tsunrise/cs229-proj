import torch
from torch import nn

class LogisticModel(nn.Module):
    def __init__(self, num_words: int, num_dep_words: int, num_categories: int):
        """
        Logistic Model using bag of words
        - Input: A vector of word counts X, where X[i] is the number of times the i-th word appears in the message.
        - Output: y = sigmoid(W@X + b)
        Here, instead of inputting a vector of word counts, we will input a vector of word indices and use EmbeddingBag to
        do the same thing. Each embedding dimension serves as logit score for a category.
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_words, num_categories, mode='sum')
        self.bias = nn.Parameter(torch.zeros(num_categories))

    def forward(self, text: torch.Tensor, text_offsets: torch.Tensor, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(text, text_offsets) + self.bias

        








        
