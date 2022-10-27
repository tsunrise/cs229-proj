import numpy as np

# hamming distance between two multi-hot encoded vectors
def hamming_distance(y_true, y_pred):
    """
    Args:
        y_true: A numpy array of shape (m, num_categories) containing the true categories.
        y_pred: A numpy array of shape (m, num_categories) containing the predicted categories.

    Returns:
        A float representing the average hamming distance between the two vectors.
    """
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=1) / y_true.shape[1])
    