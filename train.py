from model.logistic import LogisticModel
from model.nn import NNModel
from model.lstm import LSTMModel
from model.trainer import train_model
from preprocess.dataset import CrateDataset
from preprocess.prepare import CratesData
from utils.cache import cached
from utils.data import train_dev_test_split
import utils.devices as devices
import toml
import argparse
from tokenizers import Tokenizer
import model.bert_fine_tune_trainer as bert_fine_tune_trainer

TEXT_TOKENIZER_PATH = "text_tokenizer.json"
FEAT_TOKENIZER_PATH = "feat_tokenizer.json"

def load_data(force_cache_miss=False, force_download=False):
    def load():
        cratesData = CratesData(force_download=force_download)
        cratesData.remove_no_category_()
        cratesData.pre_normalize_()
        return cratesData
    crates_data = cached(load, "preprocessed_crates_train.pkl", always_miss=force_cache_miss)
    crates = [crate for crate in crates_data]
    train, dev, test = train_dev_test_split(crates)
    num_categories = len(crates_data.categories)
    return train, dev, test, num_categories

def get_standard_model(model_name, num_words, num_dep_words, num_categories):
    if model_name == "logistic":
        return LogisticModel(num_words, num_dep_words, num_categories)
    elif model_name == "nn":
        return NNModel(num_words, num_dep_words, num_categories)
    elif model_name == "lstm":
        return LSTMModel(num_words, num_dep_words, num_categories, config["hidden_size"], config["dropout_p"])
    else:
        raise ValueError("Invalid model name")

def train_standard_model(model_name, config, device, n_epochs, force_cache_miss, force_download, checkpoint=None):
    train, val, _,  num_categories = load_data(force_cache_miss, force_download)
    text_tk = Tokenizer.from_file(TEXT_TOKENIZER_PATH)
    dep_tk = Tokenizer.from_file(FEAT_TOKENIZER_PATH)
    dataset = CrateDataset(train, text_tk, dep_tk, num_categories, config["seq_len"])
    val_dataset = CrateDataset(val, text_tk, dep_tk, num_categories, config["seq_len"])

    print(f"Size of training set: {len(dataset)}")
    print(f"Size of validation set: {len(val_dataset)}")
    num_words = text_tk.get_vocab_size()
    num_dep_words = dep_tk.get_vocab_size()

    model = get_standard_model(model_name, num_words, num_dep_words, num_categories).to(device)
    train_model(model_name, model, dataset, val_dataset, config, n_epochs, device)

def train_distil_bert(model_name, config, device, n_epochs, force_cache_miss, force_download, checkpoint=None):
    train, val, _, _ = load_data(force_cache_miss, force_download)
    bert_fine_tune_trainer.train_distil_bert(config, train, val, n_epochs, device, checkpoint)
    

trainers = {"logistic": train_standard_model,
            "nn": train_standard_model,
            "lstm": train_standard_model,
            "distil_bert": train_distil_bert}


if __name__ == '__main__':
    devices.status_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=trainers.keys())
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-n","--n_epochs", type=int, default=20)
    parser.add_argument("-fc","--force_cache_miss", action="store_true")
    parser.add_argument("-fd","--force_download", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, required=False)
    args = parser.parse_args()
    trainer = trainers[args.model]
    config = toml.load("config.toml")["models"][args.model]
    trainer(args.model, config, args.device, args.n_epochs, args.force_cache_miss, args.force_download, args.checkpoint)
