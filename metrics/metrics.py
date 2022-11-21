from typing import Literal
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
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
    # for each class, the precision is the number of true positives divided by the number of true positives plus the number of false positives
    # we calculate the weighted average of the precision of each class, where the weight is the number of true positives of that class
    return label, precision_score(y_true, y_pred, average="weighted", zero_division=1)

def recall(y_true, y_pred):
    label = "recall"
    # for each class, the precision is the number of true positives divided by the number of true positives plus the number of false positives
    # we calculate the weighted average of the precision of each class, where the weight is the number of true positives of that class
    return label, recall_score(y_true, y_pred, average="weighted", zero_division=1)

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
        self.cnt = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        y_true = labels.detach().cpu().numpy()
        y_pred = logits_to_multi_hot(logits)
        for m in self.multi_label_metrics:
            k, v = m(y_true, y_pred)
            self.results.setdefault(k, 0)
            self.results[k] += v
        y_pred_labels = logits_to_labels(logits)
        for m in self.single_label_metrics:
            k, v = m(y_true, y_pred_labels)
            self.results.setdefault(k, 0)
            self.results[k] += v
        self.cnt += 1

    def get_results(self):
        return {k: v / self.cnt for k, v in self.results.items()}

    def write_to_tensorboard(self, prefix: Literal["training", "validation"] , writer: SummaryWriter, step: int, extra: dict = {}):
        for k, v in self.get_results().items():
            writer.add_scalar(f"{prefix}/{k}", v, step)
        for k, v in extra.items():
            writer.add_scalar(f"{prefix}/{k}", v, step)

def baseline_accept_rate_expected(y_true):
    """
    y_true: (num_samples, num_categories)

    p(accept) = \sum_{c=1}^k p(select c) * p(accept | select c)
              = \sum_{c=1}^k p(select c) * p(c)
    Since each p(c) ranges from 0 to 1, and p(select c) should sum to 1,
    to maximize this sum, we should select the category with the highest p(c).
    so p(accept) = \sum_{c=1}^k p(c) * I(c is the max p(c))
                 = \max_{c=1}^k p(c)
    """
    p_c = np.mean(y_true, axis=0)
    return np.max(p_c)
