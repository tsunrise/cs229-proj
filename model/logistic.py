
from typing import List
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from preprocess.utils import BagOfWordsDataset

class LogisticModel(nn.Module):
    def __init__(self, num_words: int, num_categories: int):
        """
        Logistic Model using bag of words
        - Input: A vector of word counts X, where X[i] is the number of times the i-th word appears in the message.
        - Output: y = sigmoid(W@X + b)
        Here, instead of inputting a vector of word counts, we will input a vector of word indices and use EmbeddingBag to
        do the same thing. Each embedding dimension serves as logit score for a category.
        """
        self.embedding = nn.EmbeddingBag(num_words, num_categories, mode='sum')
        self.bias = nn.Parameter(torch.zeros(num_categories))

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices, offsets) + self.bias

        
class LogisticRegression:
    def __init__(self, num_words:int, learning_rate=0.001, num_epochs=100, batch_size=128, num_categories=0, max_length=1000, device=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.max_length = max_length
        self.device = device
        self.model = LogisticModel(num_words, num_categories).to(device)
        
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, dataset: BagOfWordsDataset):
        assert dataset.num_categories == self.num_categories
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

        # train
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        writer = SummaryWriter(comment=f'logistic_{self.learning_rate}_bs_{self.batch_size}_ne_{self.num_epochs}')

        for epoch in range(self.num_epochs):
            for (indices, offsets, categories) in dataloader:
                indices = torch.from_numpy(indices).to(self.device)
                offsets = torch.from_numpy(offsets).to(self.device)
                categories = torch.from_numpy(categories).to(self.device)

                # Forward pass
                outputs = self.model(indices, offsets)
                loss = criterion(outputs, categories)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
            writer.add_scalar('Training loss', loss.item(), global_step=epoch)

        writer.close()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

        








        
