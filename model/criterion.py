import torch

from torch import nn

from preprocess.dataset import TokenizedDataset

import numpy as np
def weighted_bce_loss(train_dataset: TokenizedDataset, device):
    categories = train_dataset.categories
    category_counts = categories.sum(axis=0)
    print(f"category_counts: {category_counts}")
    category_weights = categories.shape[0] / category_counts
    category_weights = category_weights / np.min(category_weights)
    category_weights = np.minimum(category_weights, 20) 
    category_weights = torch.from_numpy(category_weights).to(device)
    print(f"category_weights: {category_weights}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=category_weights)
    return criterion

