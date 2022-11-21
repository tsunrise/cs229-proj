import torch
from torch import nn
import os
from dataclasses import dataclass
def save_snapshot(description: str, model: nn.Module, epoch: int, suffix: str = None):
    # create a snapshot directory if it doesn't exist
    if not os.path.exists("snapshots"):
        os.mkdir("snapshots")
    if suffix is None:
        suffix = f"{epoch:03d}"
    torch.save({
        "description": description,
        "model": model.state_dict(),
        "epoch": epoch,
    }, f"snapshots/{description}_{suffix}.pth")

@dataclass
class Snapshot:
    description: str
    model: nn.Module
    epoch: int

def load_snapshot(path: str):
    dic = torch.load(path)
    return Snapshot(dic["description"], dic["model"], dic["epoch"])