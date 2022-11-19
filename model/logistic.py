
from typing import List
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from preprocess.dataset import BagOfWordsDataset

class LogisticModel(nn.Module):
    def __init__(self, num_words: int, num_categories: int):
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

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices, offsets) + self.bias

        
class LogisticRegression:
    def __init__(self, num_words:int, learning_rate=0.001, num_epochs=100, batch_size=64, num_categories=0, max_length=1000, device=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.max_length = max_length
        self.device = device
        self.model = LogisticModel(num_words, num_categories).to(device)
        
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device {self.device}')
        print(self.model)

    def fit(self, train_dataset: BagOfWordsDataset, val_dataset: BagOfWordsDataset):
        assert train_dataset.num_categories == self.num_categories
        
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=val_dataset.collate_fn)
        # train
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        writer = SummaryWriter(comment=f'logistic_{self.learning_rate}_bs_{self.batch_size}_ne_{self.num_epochs}')

        for epoch in range(self.num_epochs):
            num_batches = len(dataloader)
            total_loss = 0
            for (indices, offsets, categories) in dataloader:
                indices = torch.from_numpy(indices).to(self.device)
                offsets = torch.from_numpy(offsets).to(self.device)
                categories = torch.from_numpy(categories).to(self.device)

                # Forward pass
                outputs = self.model(indices, offsets)
                loss = criterion(outputs, categories)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validate
            with torch.no_grad():
                val_loss = 0
                num_val_batches = 0
                for (indices, offsets, categories) in val_dataloader:
                    indices = torch.from_numpy(indices).to(self.device)
                    offsets = torch.from_numpy(offsets).to(self.device)
                    categories = torch.from_numpy(categories).to(self.device)

                    outputs = self.model(indices, offsets)
                    loss = criterion(outputs, categories)
                    val_loss += loss.item()
                    num_val_batches += 1

            average_loss = total_loss / num_batches
            average_val_loss = val_loss / num_val_batches
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')
            writer.add_scalar('Training loss', average_loss, global_step=epoch)
            writer.add_scalar('Validation loss', average_val_loss, global_step=epoch)

        writer.close()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

        








        
