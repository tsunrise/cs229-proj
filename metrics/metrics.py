from typing import Literal
import numpy as np
import torch

from torch.utils.tensorboard.writer import SummaryWriter

def accept_rate(labels, logits):

    """
    In this metric, if the output label is contained in the true labels, it is considered correct.
    This is not that trivial: https://crates.io/categories?sort=crates
    """

    pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return np.mean(labels[np.arange(labels.shape[0]), pred_labels])

class PerformanceTracker:
    def __init__(self, num_classes: int):
        self.tp = np.zeros(num_classes)
        self.fp = np.zeros(num_classes)
        self.fn = np.zeros(num_classes)
        self.accept_rate_sum = 0
        self.cnt = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        pred = (logits.detach().cpu().numpy() > 0).astype(int)
        labels = labels.detach().cpu().numpy()

        self.tp += np.sum(pred * labels, axis=0)
        self.fp += np.sum(pred * (1 - labels), axis=0)
        self.fn += np.sum((1 - pred) * labels, axis=0)
        self.accept_rate_sum += accept_rate(labels, logits)
        self.cnt += 1

    def get_results(self):
        weight = self.tp + self.fn
        weight /= np.sum(weight)
        
        tpfp_mask = self.tp + self.fp > 0
        precision = np.sum((self.tp[tpfp_mask] / (self.tp[tpfp_mask] + self.fp[tpfp_mask])) * weight[tpfp_mask])
        tpfn_mask = self.tp + self.fn > 0
        recall = np.sum((self.tp[tpfn_mask] / (self.tp[tpfn_mask] + self.fn[tpfn_mask])) * weight[tpfn_mask])
        return {
            "precision": precision,
            "recall": recall,
            "accept_rate": self.accept_rate_sum / self.cnt
        }

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
