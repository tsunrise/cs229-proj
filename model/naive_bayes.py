"""
Multi-label Naive Bayes classifier.
"""
import numpy as np
import tqdm

class NaiveBayes:

    def fit(self, X, y, num_categories: int):
        """
        Fit the model to the data.

        Args:
            X: A numpy array of shape (num_crates, num_words) containing the word counts for each crate.
            y: A numpy array of shape (num_crates, num_categories) containing one-hot encoded categories (can be multiple).
        """
        # prior
        phi_y = np.mean(y, axis=0)
        # laplace smoothing
        k = X.shape[1]
        # likelihood
        log_phi_xy1 = []
        log_phi_xy0 = []
        for i in tqdm.tqdm(range(num_categories), desc='Fitting model'):
            log_phi_xy1.append(np.log((np.sum(X[y[:, i] == 1], axis=0) + 1) / (np.sum(y[:, i] == 1) + k)).reshape(-1))
            log_phi_xy0.append(np.log((np.sum(X[y[:, i] == 0], axis=0) + 1) / (np.sum(y[:, i] == 0) + k)).reshape(-1))
        
        self.phi_y = phi_y
        self.log_phi_xy1 = log_phi_xy1
        self.log_phi_xy0 = log_phi_xy0
        
        self.num_categories = num_categories
        self.num_crates = X.shape[0]
        self.num_words = X.shape[1]

    def predict(self, X):
        """
        Predict the categories for each crate in X.

        Args:
            X: A numpy array of shape (m, num_words) containing the word counts for each crate.

        Returns:
            A numpy array of shape (m, num_categories) containing the predicted decision for each category.
        """

        decision = np.zeros((X.shape[0], self.num_categories))
        log_p1s = []
        log_p0s = []

        for i in tqdm.tqdm(range(self.num_categories), desc='Predicting categories'):
            if self.phi_y[i] == 0:
                log_p1s.append(np.zeros(X.shape[0]))
                log_p0s.append(np.zeros(X.shape[0]))
                decision[:, i] = 0
            else:
                log_p1 = X @ self.log_phi_xy1[i] + np.log(self.phi_y[i])
                log_p0 = X @ self.log_phi_xy0[i] + np.log(1 - self.phi_y[i])
                log_p1s.append(log_p1)
                log_p0s.append(log_p0)

                decision[:, i] = (log_p1 > log_p0).astype(int)
        
        return decision, np.array(log_p1s), np.array(log_p0s)


    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Args:
            X: A numpy array of shape (num_crates, num_words) containing the word counts for each crate.
            y: A numpy array of shape (num_crates, num_categories) containing one-hot encoded categories (can be multiple).

        Returns:
            The accuracy of the model.
        """
        ...