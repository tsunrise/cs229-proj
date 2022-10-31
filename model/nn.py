"""
NeuralNet Word bag model
"""

import numpy as np
import torch
import torch.nn as nn

from metrics.metrics import accuracy_samples

class LogisticRegression(nn.Module):
    def __init__(self, num_words, num_categories, device):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(num_words, num_categories).to(device)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.fc1(x)


class MyNet(nn.Module):
    def __init__(self, num_words, num_categories, device):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(num_words, 500).to(device)
        self.fc2 = nn.Linear(500, 500).to(device)
        self.fc3 = nn.Linear(500, num_categories).to(device)
        # self.fc3s = [nn.Linear(500, 20).to(device) for _ in range(num_categories)]
        # self.fc4 = [nn.Linear(20, 1).to(device) for _ in range(num_categories)]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        # x = self.relu(x)
        # outs = []
        # for i in range(len(self.fc3s)):
        #     a = self.fc3s[i](x)
        #     a = self.relu(a)
        #     a = self.fc4[i](a)
        #     outs.append(a)
        # return torch.cat(outs, dim=1)


class DNNModel:
    def __init__(self, num_categories, learning_rate=0.001, reg=0.0, weight_decay=0.0):
        self.num_categories = num_categories
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.reg = reg
        self.weight_decay = weight_decay
        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, model_func, X_train, y_train, X_val, y_val, epochs=10, batch_size = 64):
        """
        Fit the model to the data.

        Args:
            X: A numpy array of shape (num_crates, num_words) containing the word counts for each crate.
            y: A numpy array of shape (num_crates, num_categories) containing one-hot encoded categories (can be multiple).
        """
        self.num_crates = X_train.shape[0]
        self.num_words = X_train.shape[1]

        # shuffle X and y
        idx = np.arange(self.num_crates)
        np.random.shuffle(idx)

        self.model = model_func(self.num_words, self.num_categories, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            epoch_loss = 0
            for batch_lo in range(0, X_train.shape[0], batch_size):
                local_batch_size = min(batch_size, self.num_crates - batch_lo)
                batch_X = torch.tensor(X_train[batch_lo:batch_lo + local_batch_size], device=self.device)
                batch_y = torch.tensor(y_train[batch_lo:batch_lo + local_batch_size], device=self.device)
                output = self.model(batch_X)
                loss = self.loss(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / (X_train.shape[0])
            # evaluate on validation set
            with torch.no_grad():
                epoch_loss = 0
                for batch_lo in range(0, X_val.shape[0], batch_size):
                    local_batch_size = min(batch_size, self.num_crates - batch_lo)
                    batch_X = torch.from_numpy(X_val[batch_lo:batch_lo + local_batch_size]).float().to(self.device)
                    batch_y = torch.from_numpy(y_val[batch_lo:batch_lo + local_batch_size]).float().to(self.device)
                    output = self.model(batch_X)
                    loss = self.loss(output, batch_y)
                    epoch_loss += loss.item()
                val_loss = epoch_loss / X_val.shape[0]

            with torch.no_grad():
                y_train_pred = self.predict(X_train)
                y_val_pred = self.predict(X_val)
                _, train_hamming_dist = accuracy_samples(y_train, y_train_pred)
                train_hamming_dist = 1 - train_hamming_dist
                _, val_hamming_dist = accuracy_samples(y_val, y_val_pred)
                val_hamming_dist = 1 - val_hamming_dist
                print("Epoch: {}, Loss: {:.4f}, val_loss: {:.4f}, Hamming Distance: {:.4f}, val_hamming_dist: {:.4f}".format(epoch, train_loss, val_loss, train_hamming_dist, val_hamming_dist))

    def predict(self, X, logits=False):
        """
        Predict the categories for each crate in X.

        Args:
            X: A numpy array of shape (m, num_words) containing the word counts for each crate.

        Returns:
            A numpy array of shape (m, num_categories) containing the predicted decision for each category.
        """
        X = torch.from_numpy(X).float().to(self.device)
        decision = self.model(X).detach().cpu().numpy()
        if logits:
            return decision
        else:
            return (decision > 0).astype(int)