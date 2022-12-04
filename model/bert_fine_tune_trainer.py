from pathlib import Path
from typing import List
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import get_scheduler, DistilBertModel, DistilBertTokenizerFast
from model.dep import FeatNet
from model.loss import AsymmetricLossOptimized, weighted_bce_loss
from preprocess.dataset import BertDataset
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from metrics.metrics import PerformanceTracker
from train import FEAT_TOKENIZER_PATH
from torch.utils.data import DataLoader
from preprocess.prepare import Crate
from utils.temp import DataPaths
import toml

class DepNetPrepare(nn.Module):
    """
    Before training DistriBERT transfer learning model, we need to prepare the DepNet model and 
    use the prepared parameters to initialize the DepNet part of DistilBERT model.
    """
    def __init__(self, num_categories, num_dep_words: int, hidden_him: int, dropout_p: float = 0.1):
        super().__init__()
        self.depnet = FeatNet(num_dep_words, hidden_him, dropout_p)
        self.linear = nn.Linear(hidden_him, num_categories)

    def forward(self, deps, deps_offsets):
        deps = self.depnet(deps, deps_offsets)
        return self.linear(deps)

def prepare_depnet(train_dataloader: DataLoader, num_samples, n_epochs, num_dep_words, config: dict, criterion, device, writer = None) -> FeatNet:
    assert config["name"] == "depnet"
    model = DepNetPrepare(config["num_categories"], num_dep_words, config["hidden_dim"], config["dropout_p"])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    progress_bar = tqdm(range(n_epochs), desc="Preparing DepNet")
    for epoch in progress_bar:
        loss_total = 0
        for batch in train_dataloader:
            model.train()
            deps = batch["deps"].to(device)
            deps_offsets = batch["deps_offsets"].to(device)
            categories = batch["categories"].to(device)
            logits = model(deps, deps_offsets)
            loss = criterion(logits, categories)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += loss.item()
            if writer is not None:
                writer.add_scalar("depnet_prepare/loss", loss, epoch)
        progress_bar.set_postfix({"loss": loss_total / num_samples})
    return model.depnet  

class DistilBERTTransferLearning(nn.Module):
    BERT_HIDDEN_DIM = 768
    def __init__(self, pretrained_name: str, num_categories: int, num_dep_words, depnet_hidden_dim, depnet_dropout_p, no_dep: bool, dropout_p: float = 0.1, dropout_p_from_depnet: float = 0.5):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_from_depnet = nn.Dropout(dropout_p_from_depnet)
        self.depnet = FeatNet(num_dep_words, depnet_hidden_dim, depnet_dropout_p)
        self.output = nn.Linear(self.BERT_HIDDEN_DIM + depnet_hidden_dim, num_categories)
        if no_dep:
            self.depnet = None
            self.output = nn.Linear(self.BERT_HIDDEN_DIM, num_categories)

    def forward(self, ids, mask, deps, deps_offsets):
        X = self.bert(ids, attention_mask=mask)[0]   # type: ignore
        # pool across the sequence dimension (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        X = torch.mean(X, dim=1)
        X = self.dropout(X)
        if self.depnet is not None:
            deps = self.depnet(deps, deps_offsets)
            deps = self.dropout_from_depnet(deps)
            X = torch.cat([X, deps], dim=1)
            return self.output(X)
        else:
            return self.output(X)

def model_forward(model: DistilBERTTransferLearning, batch, device):
    ids = batch["ids"].to(device)
    mask = batch["mask"].to(device)
    deps = batch["deps"].to(device)
    deps_offsets = batch["deps_offsets"].to(device)
    return model(ids, mask, deps, deps_offsets)

def save_checkpoint(model: DistilBERTTransferLearning, epoch: int, optimizer: torch.optim.Optimizer, scheduler, path: Path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch
    }, path)

def load_checkpoint(model: DistilBERTTransferLearning, optimizer: torch.optim.Optimizer, scheduler, path: Path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch

def train_distil_bert(config, train_crates: List[Crate],val_crates: List[Crate], num_epochs: int, device, checkpoint = None):
    tokenizer = DistilBertTokenizerFast.from_pretrained(config["pretrained"])
    deps_tokenizer = Tokenizer.from_file(FEAT_TOKENIZER_PATH)

    train_dataset = BertDataset(train_crates, tokenizer, deps_tokenizer, config["max_length"], config["num_categories"])
    val_dataset = BertDataset(val_crates, tokenizer, deps_tokenizer, config["max_length"], config["num_categories"])
    
    dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=BertDataset.bert_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=BertDataset.bert_collate_fn)

    no_dep = config["no_dep"]

    model = DistilBERTTransferLearning(config["pretrained"], config["num_categories"], num_dep_words=deps_tokenizer.get_vocab_size(), depnet_hidden_dim=config["depnet"]["hidden_dim"], depnet_dropout_p=config["depnet"]["dropout_p"],
     no_dep=no_dep, dropout_p=config["depnet"]["dropout_p"], dropout_p_from_depnet=config["dropout_p"])
    model.to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(dataloader))

    if config["loss"] == "weighted_bce":
        criterion = weighted_bce_loss(train_dataset.categories.sum(axis=0), len(train_dataset), config["pos_weight_threshold"]).to(device)
    elif config["loss"] == "assymetric":
        criterion = AsymmetricLossOptimized()
    else:
        raise ValueError(f"Unknown loss {config['loss']}")
    writer = SummaryWriter(comment=f'distilbert_{config["learning_rate"]}_bs_{config["batch_size"]}_ne_{num_epochs}_{config["loss"]}')
    writer.add_text("config", toml.dumps(config))
    paths = DataPaths()

    if checkpoint is not None:
        epoch_start = load_checkpoint(model, optimizer, lr_scheduler, checkpoint)
        print(f"Loaded checkpoint {checkpoint} at epoch {epoch_start}")
    else:
        if not no_dep:
            depnet_config = config["depnet"]
            depnet = prepare_depnet(dataloader, len(train_dataset), depnet_config["n_epochs"], deps_tokenizer.get_vocab_size(), depnet_config, criterion, device, writer)
            model.depnet.load_state_dict(depnet.state_dict())
        epoch_start = 0

    for epoch in range(epoch_start, num_epochs):
        training_perf = PerformanceTracker(config["num_categories"])
        val_perf = PerformanceTracker(config["num_categories"])
        model.train()
        for batch in tqdm(dataloader, desc="Train {}".format(epoch)):
            logits = model_forward(model, batch, device)
            categories = batch["categories"].to(device)
            loss = criterion(logits, categories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            training_perf.update(logits, categories, loss.item())
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Val {}".format(epoch)):
                logits = model_forward(model, batch, device)
                categories = batch["categories"].to(device)
                loss = criterion(logits, categories)
                val_perf.update(logits, categories, loss.item())
        metrics_train = training_perf.result_str()
        metrics_val = val_perf.result_str()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train: {metrics_train}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val: {metrics_val}")

        training_perf.write_to_tensorboard("training", writer, epoch)
        val_perf.write_to_tensorboard("validation", writer, epoch)

        save_checkpoint(model, epoch, optimizer, lr_scheduler, paths.snapshots_dir / f"distilbert_checkpoint.pt")
    save_checkpoint(model, epoch, optimizer, lr_scheduler, paths.snapshots_dir / f"distilbert_{epoch}.pt")





    

