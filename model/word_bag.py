from preprocess.dataset import BagOfWordsDataset
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics import metrics
from .criterion import weighted_bce_loss

def train_word_bag_model(model_name: str, model: nn.Module, train_dataset: BagOfWordsDataset,
                         val_dataset: BagOfWordsDataset, config: dict, num_epochs: int, device=None):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=val_dataset.collate_fn)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    writer = SummaryWriter(comment=f'{model_name}_{config["learning_rate"]}_bs_{config["batch_size"]}_ne_{num_epochs}')

    for epoch in range(num_epochs):
        num_batches = len(dataloader)
        total_loss = 0
        training_perf = metrics.PerformanceTracker()
        val_perf = metrics.PerformanceTracker()

        model.train()
        for (indices, offsets, categories) in dataloader:
            indices = torch.from_numpy(indices).to(device)
            offsets = torch.from_numpy(offsets).to(device)
            categories = torch.from_numpy(categories).to(device)

            # Forward pass
            outputs = model(indices, offsets)
            loss = criterion(outputs, categories)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # performance evaluate
            training_perf.update(outputs, categories)

        # validate
        with torch.no_grad():
            model.eval()
            val_loss = 0
            num_val_batches = 0
            for (indices, offsets, categories) in val_dataloader:
                indices = torch.from_numpy(indices).to(device)
                offsets = torch.from_numpy(offsets).to(device)
                categories = torch.from_numpy(categories).to(device)

                outputs = model(indices, offsets)
                loss = criterion(outputs, categories)
                val_loss += loss.item()
                num_val_batches += 1

                val_perf.update(outputs, categories)


        average_loss = total_loss / num_batches
        average_val_loss = val_loss / num_val_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')
        training_perf = training_perf.get_results()
        val_perf = val_perf.get_results()
        print(f'Training: {training_perf}')
        print(f'Validation: {val_perf}')
        writer.add_scalar('Training loss', average_loss, global_step=epoch)
        writer.add_scalar('Validation loss', average_val_loss, global_step=epoch)
        writer.add_scalar('Training precision', training_perf['precision'], global_step=epoch)
        writer.add_scalar('Validation precision', val_perf['precision'], global_step=epoch)
        writer.add_scalar('Training recall', training_perf['recall'], global_step=epoch)
        writer.add_scalar('Validation recall', val_perf['recall'], global_step=epoch)
        writer.add_scalar('Training accept rate', training_perf['accept_rate'], global_step=epoch)
        writer.add_scalar('Validation accept rate', val_perf['accept_rate'], global_step=epoch)

        writer.close()
