from typing import List, Optional
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

def train_tokenizer(crates: List[Crate], num_words: int = 25000, save_path: Optional[str] = None):
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

    if save_path is not None:
        tk.save(save_path)

    return tk

def train_tokenizer_main(num_words: int = 25000, save_path: str = "tokenizer.json", force_download: bool = False, cache_readme: bool = False):
    # get crates
    def get_data():
        cratesData = CratesData(force_download=force_download)
        cratesData.remove_no_category_()
        cratesData.process_readme_()
        return cratesData
    
    cratesData = cached(get_data, "crates_for_tokenizer.pkl", always_miss = not cache_readme)
    crates = list(cratesData.all_crates())

    # train tokenizer
    return train_tokenizer(crates, num_words, save_path)






