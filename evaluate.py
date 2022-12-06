from tokenizers import Tokenizer
from model.bert_fine_tune_trainer import get_bert_model
from model.trainer import get_collate_fn, model_forward
import torch
import torch.utils.data
from metrics.metrics import PerformanceTracker
from preprocess.dataset import BertDataset, CrateDataset
from train import FEAT_TOKENIZER_PATH, TEXT_TOKENIZER_PATH, get_standard_model, load_data
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
import model.bert_fine_tune_trainer as bert_fine_tune_trainer
import toml
def evaluate_on_dataset(model, dataset, device, config):
    collate_fn = get_collate_fn(model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    perf = PerformanceTracker(config["num_categories"])
    model.eval()
    torch.set_grad_enabled(False)
    output_logits = []
    true_labels = []
    for batch in tqdm(dataloader):
        categories = batch["categories"].to(device)
        outputs = model_forward(model, batch, device)
        perf.update(outputs, categories, 0)
        output_logits.append(outputs.detach().cpu().numpy())
        true_labels.append(categories.detach().cpu().numpy())
    return perf, output_logits, true_labels

def get_results(perf: PerformanceTracker, output_logits, true_labels):
    return {
        "perf": perf.get_results()
        # TODO: add AUC
        # TODO: add ROC curve
    }

def summarize_results(train_results, val_results, test_results, config):
    return {
        "config": config,
        "train": train_results,
        "val": val_results,
        "test": test_results
    }


def evaluate_standard_model(model_name, config, device, force_cache_miss, force_download, checkpoint):
    train, val, test, num_categories = load_data(force_cache_miss, force_download)
    text_tk = Tokenizer.from_file(TEXT_TOKENIZER_PATH)
    dep_tk = Tokenizer.from_file(FEAT_TOKENIZER_PATH)
    train_dataset = CrateDataset(train, text_tk, dep_tk, num_categories, config["seq_len"])
    val_dataset = CrateDataset(val, text_tk, dep_tk, num_categories, config["seq_len"])
    test_dataset = CrateDataset(test, text_tk, dep_tk, num_categories, config["seq_len"])

    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    num_words = text_tk.get_vocab_size()
    num_dep_words = dep_tk.get_vocab_size()
    model = get_standard_model(model_name, num_words, num_dep_words, num_categories).to(device)
    model.load_state_dict(torch.load(checkpoint))

    print("Evaluating on train")
    train_perf, train_logits, train_labels = evaluate_on_dataset(model, train_dataset, device, config)
    print("Evaluating on val")
    val_perf, val_logits, val_labels = evaluate_on_dataset(model, val_dataset, device, config)
    print("Evaluating on test")
    test_perf, test_logits, test_labels = evaluate_on_dataset(model, test_dataset, device, config)

    train_results = get_results(train_perf, train_logits, train_labels)
    val_results = get_results(val_perf, val_logits, val_labels)
    test_results = get_results(test_perf, test_logits, test_labels)

    return summarize_results(train_results, val_results, test_results, config)

def evaluate_on_bert_dataset(model, dataset: BertDataset, device, config):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.bert_collate_fn)
    perf = PerformanceTracker(config["num_categories"])
    model.eval()
    torch.set_grad_enabled(False)
    output_logits = []
    true_labels = []
    for batch in tqdm(dataloader):
        categories = batch["categories"].to(device)
        outputs = bert_fine_tune_trainer.model_forward(model, batch, device)
        perf.update(outputs, categories, 0)
        output_logits.append(outputs.detach().cpu().numpy())
        true_labels.append(categories.detach().cpu().numpy())
    return perf, output_logits, true_labels

def evaluate_bert_model(model_name, config, device, force_cache_miss, force_download, checkpoint):
    tokenizer = DistilBertTokenizerFast.from_pretrained(config["pretrained"])
    deps_tokenizer = Tokenizer.from_file(FEAT_TOKENIZER_PATH)
    train, val, test, _ = load_data(force_cache_miss, force_download)
    train_dataset = BertDataset(train, tokenizer, deps_tokenizer, config["max_length"], config["num_categories"])
    val_dataset = BertDataset(val, tokenizer, deps_tokenizer, config["max_length"], config["num_categories"])
    test_dataset = BertDataset(test, tokenizer, deps_tokenizer, config["max_length"], config["num_categories"])
    model = get_bert_model(config, deps_tokenizer.get_vocab_size(), device)
    model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    print("Evaluating on train")
    train_perf, train_logits, train_labels = evaluate_on_bert_dataset(model, train_dataset, device, config)
    print("Evaluating on val")
    val_perf, val_logits, val_labels = evaluate_on_bert_dataset(model, val_dataset, device, config)
    print("Evaluating on test")
    test_perf, test_logits, test_labels = evaluate_on_bert_dataset(model, test_dataset, device, config)

    train_results = get_results(train_perf, train_logits, train_labels)
    val_results = get_results(val_perf, val_logits, val_labels)
    test_results = get_results(test_perf, test_logits, test_labels)

    return summarize_results(train_results, val_results, test_results, config)



evaluator = {
    "logistic": evaluate_standard_model,
    "nn": evaluate_standard_model,
    "distil_bert": evaluate_bert_model,
}

if __name__ == "__main__":
    CONFIG_PATH = "config.toml"
    import argparse
    from torch.utils.tensorboard.writer import SummaryWriter
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("-fc", "--force-cache-miss", action="store_true", help="Force cache miss")
    parser.add_argument("-fd", "--force-download", action="store_true", help="Force download")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()
    config = toml.load(CONFIG_PATH)

    summary_writer = SummaryWriter(f"runs/eval/{args.model}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    result = evaluator[args.model](args.model, config["models"][args.model], args.device, args.force_cache_miss, args.force_download, args.checkpoint)
    print(result)
    summary_writer.add_text("results", toml.dumps(result))
    summary_writer.close()

