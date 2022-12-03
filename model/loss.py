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
    
# 
class AsymmetricLossOptimized(nn.Module):
    """
    Source: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
     @misc{benbaruch2020asymmetric, 
        title={Asymmetric Loss For Multi-Label Classification}, 
        author={Emanuel Ben-Baruch and Tal Ridnik and Nadav Zamir and Asaf Noy and Itamar Friedman and Matan Protter and Lihi Zelnik-Manor}, 
        year={2020}, 
        eprint={2009.14119},
        archivePrefix={arXiv}, 
        primaryClass={cs.CV} }
    """
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

