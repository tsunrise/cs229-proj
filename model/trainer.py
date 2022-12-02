from model.logistic import LogisticModel
from model.loss import weighted_bce_loss
from model.nn import NNModel
from preprocess.dataset import CrateDataset
from torch import nn
import torch
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from metrics import metrics

def get_collate_fn(model):
    if isinstance(model, NNModel):
        return CrateDataset.word_bag_collate_fn
    elif isinstance(model, LogisticModel):
        return CrateDataset.word_bag_collate_fn
    else:
        raise NotImplementedError()

def model_forward(model: nn.Module, batch, device):
    if isinstance(model, NNModel) or isinstance(model, LogisticModel):
        text = batch["text"].to(device)
        text_offsets = batch["text_offsets"].to(device)
        deps = batch["deps"].to(device)
        deps_offsets = batch["deps_offsets"].to(device)
        return model(text, text_offsets, deps, deps_offsets)
    else:
        raise NotImplementedError()

def train_model(model_name: str, model: nn.Module, train_dataset: CrateDataset,
                         val_dataset: CrateDataset, config: dict, num_epochs: int, device=None):
    # model check
    if not (isinstance(model, NNModel) or isinstance(model, LogisticModel)):
        raise ValueError("model must be a LogisticModel or NNModel")
    
    collate_fn = get_collate_fn(model)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    baseline_ac_rate_expected = metrics.baseline_accept_rate_expected(train_dataset.categories)
    print(f"baseline accept rate: {baseline_ac_rate_expected}")

    criterion = weighted_bce_loss(train_dataset.categories.sum(axis=0), len(train_dataset), pos_weight_threshold=config["pos_weight_threshold"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    writer = SummaryWriter(comment=f'{model_name}_{config["learning_rate"]}_bs_{config["batch_size"]}_ne_{num_epochs}', flush_secs=30)

    for epoch in range(num_epochs):
        training_perf = metrics.PerformanceTracker(config["num_categories"])
        val_perf = metrics.PerformanceTracker(config["num_categories"])

        model.train()
        for batch in dataloader:
            categories = batch["categories"].to(device)

            outputs = model_forward(model, batch, device)
            loss = criterion(outputs, categories)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # performance evaluate
            training_perf.update(outputs, categories, loss.item())

        # validate
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                categories = batch["categories"].to(device)

                outputs = model_forward(model, batch, device)
                loss = criterion(outputs, categories)

                val_perf.update(outputs, categories, loss.item())

        metrics_train = training_perf.result_str()
        metrics_val = val_perf.result_str()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train: {metrics_train}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val: {metrics_val}')

        training_perf.write_to_tensorboard("training", writer, epoch)
        val_perf.write_to_tensorboard("validation", writer, epoch)
