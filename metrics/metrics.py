import numpy as np
import torch

def logits_to_multi_hot(logits):
    if isinstance(logits, list):
        logits = np.array(logits)
    elif isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    elif isinstance(logits, np.ndarray):
        pass
    else:
        raise Exception("logits_to_multi_hot: unsupported type for logits")
    return (logits > 0).astype(int)

def logits_to_labels(logits):
    if isinstance(logits, list):
        logits = np.array(logits)
    elif isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    elif isinstance(logits, np.ndarray):
        pass
    else:
        raise Exception("logits_to_labels: unsupported type for logits")
    return np.argmax(logits, axis=1)

def precision(y_true, y_pred):
    label = "precision"
    # for each sample, the precision is the number of true positives divided by the number of predicted positives
    denom = np.sum(y_pred, axis=1)
    mask = denom > 0
    precision = np.sum(y_true * y_pred, axis=1)[mask] / denom[mask]
    return label, np.mean(precision) if len(precision) > 0 else 0

def recall(y_true, y_pred):
    label = "recall"
    # for each sample, the recall is the number of true positives divided by the number of true positives plus the number of false negatives
    denom = np.sum(y_true, axis=1)
    mask = denom > 0
    recall = np.sum(y_true * y_pred, axis=1)[mask] / denom[mask]
    return label, np.mean(recall) if len(recall) > 0 else 0

def accept_rate(y_true, y_pred_labels):
    """
    y_pred_labels: (batch_size,) index of the predicted label
    In this metric, if the output label is contained in the true labels, it is considered correct.
    This is not that trivial: https://crates.io/categories?sort=crates
    """
    label = "accept_rate"
    return label, np.mean(y_true[np.arange(y_true.shape[0]), y_pred_labels])

class PerformanceTracker:
    def __init__(self):
        self.multi_label_metrics = [precision, recall]
        self.single_label_metrics = [accept_rate]
        self.results = {}

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        y_true = labels.detach().cpu().numpy()
        y_pred = logits_to_multi_hot(logits)
        for m in self.multi_label_metrics:
            k, v = m(y_true, y_pred)
            self.results.setdefault(k, []).append(v)
        y_pred_labels = logits_to_labels(logits)
        for m in self.single_label_metrics:
            k, v = m(y_true, y_pred_labels)
            self.results.setdefault(k, []).append(v)

    def get_results(self):
        return {k: np.mean(v) for k, v in self.results.items()}