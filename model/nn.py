import torch
from torch import nn
from torch.nn import functional as F

from model.dep import FeatNet

class NNModel(nn.Module):
    def __init__(self, num_words: int, num_feat_words: int, num_categories: int):
        """
        Neural Net Model using bag of words
        - Input: A vector of word counts X, where X[i] is the number of times the i-th word appears in the message.
        - Output: logits score for a category.
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_words, 512, mode='sum')
        self.depnet = FeatNet(num_feat_words, 512, 0.55)
        self.dropout = nn.Dropout(0.55)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_categories)


    def forward(self, text: torch.Tensor, text_offsets: torch.Tensor, feats: torch.Tensor, feats_offsets: torch.Tensor) -> torch.Tensor:
        # TODO: deps is not used yet
        X = self.embedding(text, text_offsets)
        X_dep = self.depnet(feats, feats_offsets)

        X = torch.cat((X, X_dep), dim=1)
        X = F.leaky_relu(X)
        X = self.dropout(X)

        X_rc = self.fc1(X)
        X_rc = F.leaky_relu(X_rc)
        X = X + X_rc
        X = self.dropout(X)
        X = self.fc2(X)
        return X

# class Test(nn.Module):
#     def __init__(self):
#         """
#         Neural Net Model using bag of words
#         - Input: A vector of word counts X, where X[i] is the number of times the i-th word appears in the message.
#         - Output: logits score for a category.
#         """
#         super().__init__()
#         self.embedding = nn.EmbeddingBag(num_words, 512, mode='sum')
#         self.depnet = DepNet(num_dep_words, 512, 0.55)
#         self.fc1 = nn.Linear(1024, 1024)

#     def forward(self, text: torch.Tensor, text_offsets: torch.Tensor, deps: torch.Tensor, deps_offsets: torch.Tensor) -> torch.Tensor:
#         # TODO: deps is not used yet
#         X = self.embedding(text, text_offsets)
#         X_dep = self.depnet(deps, deps_offsets)

#         X = torch.cat((X, X_dep), dim=1)
#         X = F.leaky_relu(X)

#         X_rc = self.fc1(X)
#         X_rc = F.leaky_relu(X_rc)
#         X = X + X_rc
#         return X

# nn_model = NNModel(num_words, num_dep_words, num_categories);
# train nn_model or use existing parameters
# test = Test()
# set
# test.embedding = nn_model.embedding;
# ...