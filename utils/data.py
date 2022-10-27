import numpy as np

def train_dev_split(X, y, train_ratio=0.8, seed=0):
    """
    Split the data into training and development sets.

    Args:
        X: A numpy array of shape (m, d1).
        y: A numpy array of shape (m, d2).
        train_ratio: The ratio of training data to total data.

    Returns:
        X_train: A numpy array of shape (m_train, d1).
        y_train: A numpy array of shape (m_train, d2).
        X_dev: A numpy array of shape (m_dev, d1).
        y_dev: A numpy array of shape (m_dev, d2).
    """
    np.random.seed(seed)
    m = X.shape[0]
    m_train = int(m * train_ratio)
    indices = np.random.permutation(m)
    
    X_train = X[indices[:m_train]]
    y_train = y[indices[:m_train]]
    X_dev = X[indices[m_train:]]
    y_dev = y[indices[m_train:]]
    return X_train, y_train, X_dev, y_dev