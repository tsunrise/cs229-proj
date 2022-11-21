import torch
from torch import nn
import os
from dataclasses import dataclass
def save_snapshot(description: str, model: nn.Module, epoch: int):
    # create a snapshot directory if it doesn't exist
    if not os.path.exists("snapshots"):
        os.mkdir("snapshots")

    torch.save({
        "description": description,
        "model": model.state_dict(),
        "epoch": epoch,
    }, f"snapshots/{description}_{epoch}.pth")

@dataclass
class Snapshot:
    description: str
    model: nn.Module
    epoch: int

def load_snapshot(path: str):
    dic = torch.load(path)
    return Snapshot(dic["description"], dic["model"], dic["epoch"])