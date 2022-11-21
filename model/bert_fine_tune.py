from typing import List
from tqdm import tqdm
from transformers import get_scheduler, DistilBertModel, DistilBertTokenizerFast
from preprocess.dataset import BertDataset
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from metrics.metrics import PerformanceTracker
from utils import snapshots

from preprocess.prepare import Crate

class DistilBERTFineTune(nn.Module):
    def __init__(self, pretrained_name: str, num_categories: int):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, num_categories)

    def forward(self, ids, mask):
        X = self.bert(ids, attention_mask=mask)[0]
        # pool across the sequence dimension (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        X = torch.mean(X, dim=1)
        X = self.dropout(X)
        return self.output(X)


def train_distil_bert(model_name, model, config, train_crates: List[Crate],val_crates: List[Crate], num_epochs: int, device):
    tokenizer = DistilBertTokenizerFast.from_pretrained(config["pretrained"])

    train_dataset = BertDataset(train_crates, tokenizer, config["max_length"], config["num_categories"])
    val_dataset = BertDataset(val_crates, tokenizer, config["max_length"], config["num_categories"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=BertDataset.bert_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=BertDataset.bert_collate_fn)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(dataloader))

    criterion = torch.nn.BCEWithLogitsLoss()
    writer = SummaryWriter(comment=f'{model_name}_{config["learning_rate"]}_bs_{config["batch_size"]}_ne_{num_epochs}')
    for epoch in range(num_epochs):
        num_batches = len(dataloader)
        total_loss = 0
        training_perf = PerformanceTracker()
        val_perf = PerformanceTracker()

        model.train()
        progressbar = tqdm(total=num_batches, desc=f"Training")
        for item in dataloader:
            ids = item["ids"].to(device)
            mask = item["mask"].to(device)
            # token_type_ids = item["token_type_ids"].to(device) #TODO: no need
            categories = torch.from_numpy(item["categories"]).to(device)

            # Forward pass
            outputs = model(ids, mask)
            loss = criterion(outputs, categories)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # performance evaluate
            training_perf.update(outputs, categories)
            progressbar.update(1)
            progressbar.set_postfix(loss=round(loss.item(), 4), acceptance=round(training_perf.get_results()["accept_rate"], 4))
        progressbar.close()

        # validate
        with torch.no_grad():
            model.eval()
            val_loss = 0
            num_val_batches = 0
            for item in tqdm(val_dataloader, total=num_batches, desc=f"Val Epoch {epoch}"):
                ids = item["ids"].to(device)
                mask = item["mask"].to(device)
                categories = torch.from_numpy(item["categories"]).to(device)

                outputs = model(ids, mask)
                loss = criterion(outputs, categories)
                val_loss += loss.item()
                num_val_batches += 1

                val_perf.update(outputs, categories)


        average_loss = total_loss / num_batches
        average_val_loss = val_loss / num_val_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')
        training_perf.write_to_tensorboard("training", writer, epoch, {"loss": average_loss})
        val_perf.write_to_tensorboard("validation", writer, epoch, {"loss": average_val_loss})

        if epoch % 50 == 0:
            snapshots.save_snapshot("distilbert", model, epoch)

        print(f'Training: {training_perf.get_results()}')
        print(f'Validation: {val_perf.get_results()}')

    writer.close()





    

