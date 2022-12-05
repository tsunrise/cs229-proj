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
        # y = sigmoid(W@(X_text || X_deps) + b)
        self.embedding = nn.EmbeddingBag(num_words + num_dep_words, num_categories, mode="sum")
        self.bias = nn.Parameter(torch.zeros(num_categories))
        self.num_words = num_words

    def forward(self, text: torch.Tensor, text_offsets: torch.Tensor, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
        """
        :param text: Tensor of word indices
        :param text_offsets: Tensor of offsets for each message
        :param deps: Tensor of dependency indices
        :param deps_offsets: Tensor of offsets for each message
        :return: Tensor of shape (batch_size, num_categories)
        """
        text_embed = self.embedding(text, text_offsets)
        deps_embed = self.embedding(deps + self.num_words, deps_offsets)
        return text_embed + deps_embed + self.bias


        








        
