from typing import List, Optional
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import copy

from preprocess.prepare import Crate, CratesData
from utils.data import train_dev_split
from utils.cache import cached
# tokens
UNKNOWN_TOKEN = "[UNK]"
SPECIAL_TOKENS = [UNKNOWN_TOKEN]

def train_tokenizer(crates: List[Crate], num_words: int = 25000, num_deps = 2000, text_path: str = "text_tokenizer.json", dep_path: str = "dep_tokenizer.json"):
    # Train Text Tokenizer
    tk = Tokenizer(models.WordPiece(unk_token=UNKNOWN_TOKEN))
    tk.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size = num_words, special_tokens = SPECIAL_TOKENS)

    def get_training_corpus():
        for i in range(0, len(crates), 1024):
            corpus = []
            for crate in crates[i:i+1024]:
                corpus.append(crate.description)
                corpus.append(crate.readme)
            yield corpus
    
    tk.train_from_iterator(get_training_corpus(), trainer)

    tk.decoder = decoders.WordPiece(prefix="##")

    tk.save(text_path)

    # Train Dep Tokenizer
    dtk = Tokenizer(models.WordLevel(unk_token=UNKNOWN_TOKEN))
    # seperate by white space
    dtk.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size = num_deps, special_tokens = SPECIAL_TOKENS)

    def get_training_corpus():
        for i in range(0, len(crates), 1024):
            corpus = []
            for crate in crates[i:i+1024]:
                corpus.append(" ".join(crate.dependency))
            yield corpus

    dtk.train_from_iterator(get_training_corpus(), trainer)
    dtk.save(dep_path)
    return tk, dtk

def get_tokenizer(text_path: str = "text_tokenizer.json", dep_path: str = "dep_tokenizer.json"):
    tk = Tokenizer.from_file(text_path)
    dtk = Tokenizer.from_file(dep_path)
    return tk, dtk

def train_tokenizer_main(num_words: int = 25000, num_deps=2000, force_download: bool = False):
    # get crates
    cratesData = CratesData(force_download=force_download)
    # copy cratesData
    cratesDateUnsupervised = copy.deepcopy(cratesData)
    cratesData.remove_no_category_()
    cratesDateUnsupervised.remove_with_category_()
    print(f"Number of crates with category: {len(cratesData)}")
    print(f"Number of crates without category: {len(cratesDateUnsupervised)}")
    cratesData.process_readme_()
    cratesDateUnsupervised.process_readme_()

    # exclude validation set
    crates_train = list(cratesData.all_crates())
    crates_train, _ = train_dev_split(crates_train, train_ratio=0.8, seed=0)

    crates = list(cratesData.all_crates()) + list(cratesDateUnsupervised.all_crates())

    # train tokenizer
    return train_tokenizer(crates, num_words, num_deps)






