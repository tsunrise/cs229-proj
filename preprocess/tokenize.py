from typing import List
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from preprocess.prepare import Crate, CratesData
from utils.cache import cached

# tokens
UNKNOWN_TOKEN = "[UNK]"
SPECIAL_TOKENS = [UNKNOWN_TOKEN]

class MyTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def from_file(path: str = "tokenizer.json"):
        tokenizer = Tokenizer.from_file(path)
        return MyTokenizer(tokenizer)

    def __call__(self, text: str):
        return self.tokenizer.encode(text).ids

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def num_words(self):
        return self.tokenizer.get_vocab_size()

    def encode_crates(self, crates: List[Crate], max_length: int):
        text = [" ".join([crate.name, "description: ", crate.description, "readme: ", crate.readme][:max_length]) for crate in crates]
        return [code.ids for code in self.tokenizer.encode_batch(text)]



def train_tokenizer(num_words: int = 25000, save_path: str = "tokenizer.json", force_download: bool = False, cache_readme: bool = False):
    tk = Tokenizer(models.WordPiece(unk_token=UNKNOWN_TOKEN))
    tk.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size = num_words, special_tokens = SPECIAL_TOKENS)

    # get crates
    def get_data():
        cratesData = CratesData(force_download=force_download)
        cratesData.remove_no_category_()
        cratesData.process_readme_()
        return cratesData
    
    cratesData = cached(get_data, "crates_for_tokenizer.pkl", always_miss = not cache_readme)
    crates = list(cratesData.all_crates())
    def get_training_corpus():
        for i in range(0, len(crates), 1024):
            corpus = []
            for crate in crates[i:i+1024]:
                corpus.append(crate.description)
                corpus.append(crate.readme)
            yield corpus
    
    tk.train_from_iterator(get_training_corpus(), trainer)

    tk.decoder = decoders.WordPiece(prefix="##")

    tk.save(save_path)

    return MyTokenizer(tk)







