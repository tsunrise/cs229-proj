from typing import Optional
import torch
from torch import nn
import os
from dataclasses import dataclass
def save_snapshot(description: str, model: nn.Module, epoch: int, suffix: str = None, optimizer=None, scheduler=None):
    # create a snapshot directory if it doesn't exist
    if not os.path.exists("snapshots"):
        os.mkdir("snapshots")
    if suffix is None:
        suffix = f"{epoch:03d}"
    torch.save({
        "description": description,
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None
    }, f"snapshots/{description}_{suffix}.pth")

@dataclass
class Snapshot:
    description: str
    model: nn.Module
    epoch: int
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]

def load_snapshot(path: str):
    dic = torch.load(path)
    return Snapshot(dic["description"], dic["model"], dic["epoch"], dic["optimizer"], dic["scheduler"])