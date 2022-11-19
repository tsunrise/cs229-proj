from model.logistic import LogisticRegression
from preprocess.dataset import BagOfWordsDataset
from preprocess.prepare import CratesData
from preprocess.tokenize import MyTokenizer
from utils.cache import cached
from utils.data import train_dev_split
import utils.devices as devices
import toml
import argparse


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

def train_logistic(device, n_epochs, force_cache_miss, force_download):
    # TODO: tokenizer might be overfitting
    config = toml.load('config.toml')
    config = config["models"]["logistic"]
    tokenizer = MyTokenizer.from_file(config["tokenizer"])
    train, val, num_categories = load_data(force_cache_miss, force_download)
    dataset = BagOfWordsDataset(train, tokenizer, config["max_length"], num_categories)
    val_dataset = BagOfWordsDataset(val, tokenizer, config["max_length"], num_categories)
    model = LogisticRegression(tokenizer.num_words(),
     learning_rate=config["learning_rate"], num_epochs=n_epochs,batch_size=config["batch_size"],
      num_categories=num_categories,max_length=config["max_length"], device=device)
    model.fit(dataset, val_dataset)
    model.save(config["name"] + ".pth")

models = {"logistic": train_logistic}

if __name__ == '__main__':
    devices.status_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=models.keys())
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-n","--n_epochs", type=int, default=20)
    parser.add_argument("-fc","--force_cache_miss", action="store_true")
    parser.add_argument("-fd","--force_download", action="store_true")
    args = parser.parse_args()
    model = models[args.model]
    model(args.device, args.n_epochs, args.force_cache_miss, args.force_download)
