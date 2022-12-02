from typing import Optional
from torch import nn
import numpy as np
import torch

# TODO: in final report, report the effect of MAX_WEIGHT on precision and recall

def weighted_bce_loss(categories_pos_count: np.ndarray, num_samples: int, pos_weight_threshold: Optional[float] = None):
    """
    Weighted BCE loss
    """
    if pos_weight_threshold is None:
        return nn.BCEWithLogitsLoss()
    categories_pos_count = categories_pos_count.astype(np.float32)
    categories_pos_count = np.clip(categories_pos_count, 1, num_samples - 1)
    categories_neg_count = num_samples - categories_pos_count
    categories_pos_weight = categories_neg_count / categories_pos_count
    categories_pos_weight = np.clip(categories_pos_weight, 0, pos_weight_threshold)
    categories_pos_weight = torch.tensor(categories_pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=categories_pos_weight)
    

