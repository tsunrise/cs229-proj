from model.logistic import LogisticModel
from model.nn import NNModel
from model.word_bag import train_word_bag_model
from preprocess.dataset import BagOfWordsDataset
from preprocess.prepare import CratesData
from preprocess.tokenize import MyTokenizer
from utils.cache import cached
from utils.data import train_dev_split
import utils.devices as devices
import toml
import argparse
from preprocess.tokenize import train_tokenizer

def load_data(force_cache_miss=False, force_download=False):
    def load():
        cratesData = CratesData(force_download=force_download)
        cratesData.remove_no_category_()
        cratesData.process_readme_()
        return cratesData
    crates_data = cached(load, "preprocessed_crates_train.pkl", always_miss=force_cache_miss)
    crates = [crate for crate in crates_data]
    train, dev = train_dev_split(crates, train_ratio=0.8, seed=0)
    num_categories = len(crates_data.categories)
    return train, dev, num_categories

def train_word_bag(config, model, model_name, device, n_epochs, force_cache_miss, force_download, tokenizer_path=None):
    train, val, num_categories = load_data(force_cache_miss, force_download)
    if tokenizer_path is not None:
        tokenizer = MyTokenizer.from_file(tokenizer_path)
    else:
        tokenizer = train_tokenizer(train, num_words=config["num_words"])
    dataset = BagOfWordsDataset(train, tokenizer, config["max_length"], num_categories)
    val_dataset = BagOfWordsDataset(val, tokenizer, config["max_length"], num_categories)
    train_word_bag_model(model_name, model, dataset, val_dataset, config, n_epochs, device)

def train_logistic(device, n_epochs, force_cache_miss, force_download, tokenizer_path=None):
    config = toml.load("config.toml")["models"]["logistic"]
    model = LogisticModel(config["num_words"], config["num_categories"]).to(device)
    model.train()
    train_word_bag(config, model, "logistic", device, n_epochs, force_cache_miss, force_download, tokenizer_path)

def train_nn(device, n_epochs, force_cache_miss, force_download, tokenizer_path=None):
    config = toml.load("config.toml")["models"]["nn"]
    model = NNModel(config["num_words"], config["num_categories"]).to(device)
    model.train()
    train_word_bag(config, model, "nn", device, n_epochs, force_cache_miss, force_download, tokenizer_path)

trainers = {"logistic": train_logistic, "nn": train_nn}

if __name__ == '__main__':
    devices.status_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=trainers.keys())
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-n","--n_epochs", type=int, default=20)
    parser.add_argument("-fc","--force_cache_miss", action="store_true")
    parser.add_argument("-fd","--force_download", action="store_true")
    parser.add_argument("-t", "--tokenizer", type=str, required=False)
    args = parser.parse_args()
    trainer = trainers[args.model]
    trainer(args.device, args.n_epochs, args.force_cache_miss, args.force_download, args.tokenizer)
