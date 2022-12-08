import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from model.bert_fine_tune_trainer import get_bert_model
from model.trainer import get_collate_fn, model_forward
import torch
import torch.utils.data
import preprocess.prepare
import utils.cache
from metrics.metrics import PerformanceTracker
from preprocess.dataset import BertDataset, CrateDataset
from train import FEAT_TOKENIZER_PATH, TEXT_TOKENIZER_PATH, get_standard_model, load_data
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
import model.bert_fine_tune_trainer as bert_fine_tune_trainer
import toml
import time

def get_category_label():
    dataset = preprocess.prepare.CratesData()
    return dataset.categories

def get_results(perf: PerformanceTracker, output_logits, true_labels, fig_prefix):
    category_labels = utils.cache.cached(get_category_label, "category_labels.pkl")

    actual = np.concatenate(true_labels)
    predict = np.concatenate(output_logits)
    num_pos_examples = np.sum(actual)

    # Compute PRC curve and AUC for each class
    precision = dict()
    recall = dict()
    auprc = np.zeros(len(actual[0]))
    auprc_weighted = 0

    for i in range(len(actual[0])):
        precision[i], recall[i], _ = metrics.precision_recall_curve(actual[:,i], predict[:,i])
        auprc[i] = metrics.auc(recall[i], precision[i])
        
        weight = np.sum(actual[:,i]) / num_pos_examples
        auprc_weighted += 0 if np.isnan(auprc[i]) else auprc[i] * weight
    
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(actual.ravel(), predict.ravel())
    auprc_micro = metrics.auc(recall["micro"], precision["micro"])
    
    # Plot PRC curves for top three AUC and weighted
    plt.figure()
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    
    max_indices = (-auprc).argsort()[:5]
    plt.plot(recall["micro"], precision["micro"], label='Micro Avg (area=%0.2f)' % auprc_micro)
    for i in max_indices:
        plt.plot(recall[i], precision[i], label='%s (area=%0.2f)' % (category_labels.get_label_name(i), auprc[i]))
    plt.legend(loc="lower right")
    plt.savefig(f'visuals/{fig_prefix}{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
    time.sleep(2)

    return {
        "perf": perf.get_results(),
        "auprc": np.nansum(auprc_weighted)
    }

def summarize_results(train_results, val_results, test_results, config):
    return {
        "config": config,
        "train": train_results,
        "val": val_results,
        "test": test_results
    }

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

    train_results = get_results(train_perf, train_logits, train_labels, model_name+"_train_")
    val_results = get_results(val_perf, val_logits, val_labels, model_name+"_val_")
    test_results = get_results(test_perf, test_logits, test_labels, model_name+"_test_")

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

    train_results = get_results(train_perf, train_logits, train_labels, model_name+"_train_")
    val_results = get_results(val_perf, val_logits, val_labels, model_name+"_val_")
    test_results = get_results(test_perf, test_logits, test_labels, model_name+"_test_")

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

