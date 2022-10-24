"""
NeuralNet Word bag model
"""

import numpy as np
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, num_words, num_categories):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(num_words, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, num_categories)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class DNNModel:
    def __init__(self, num_categories, learning_rate=0.001):
        self.num_categories = num_categories
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=10):
        """
        Fit the model to the data.

        Args:
            X: A numpy array of shape (num_crates, num_words) containing the word counts for each crate.
            y: A numpy array of shape (num_crates, num_categories) containing one-hot encoded categories (can be multiple).
        """
        self.num_crates = X.shape[0]
        self.num_words = X.shape[1]

        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        self.model = MyNet(self.num_words, self.num_categories).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            epoch_loss = 0
            total_crates = 0
            for batch_lo in range(0, self.num_crates, 64):
                local_batch_size = min(64, self.num_crates - batch_lo)
                batch_X = X[batch_lo:batch_lo + local_batch_size]
                batch_y = y[batch_lo:batch_lo + local_batch_size]
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                total_crates += local_batch_size
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss / total_crates))

    def predict(self, X):
        """
        Predict the categories for each crate in X.

        Args:
            X: A numpy array of shape (m, num_words) containing the word counts for each crate.

        Returns:
            A numpy array of shape (m, num_categories) containing the predicted decision for each category.
        """
        X = torch.from_numpy(X).float().to(self.device)
        decision = self.model(X).detach().cpu().numpy()
        return decision

    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Args:
            X: A numpy array of shape (m, num_words) containing the word counts for each crate.
            y: A numpy array of shape (m, num_categories) containing the one-hot encoded categories (can be multiple).

        Returns:
            A float representing the accuracy of the model.
        """
        raise NotImplementedError()