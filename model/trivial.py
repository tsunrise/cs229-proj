# trivial models for testing
import numpy as np

class AlwaysFalseModel:
    def __init__(self, num_categories):
        self.num_categories = num_categories
    
    def predict(self, X):
        return np.zeros((X.shape[0], self.num_categories))

class RandomLabelingModel:
    def __init__(self, num_categories):
        self.num_categories = num_categories
    
    def predict(self, X):
        return np.random.randint(2, size=(X.shape[0], self.num_categories))

class RandomAssigningModel:
    def __init__(self, num_categories):
        self.num_categories = num_categories
    
    def predict(self, X):
        assignment = np.zeros((X.shape[0], self.num_categories))
        indices = np.random.randint(0, self.num_categories, X.shape[0])
        assignment[np.arange(X.shape[0]), indices] = 1
        return assignment