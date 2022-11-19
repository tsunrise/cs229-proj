# Generate Tokenizer
from preprocess import tokenize
import argparse
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_words', type=int, default=25000)
    parser.add_argument('--save_path', type=str, default="tokenizer.json")
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--cache_readme', action='store_true')
    args = parser.parse_args()
    tokenize.train_tokenizer_main(num_words=args.num_words, save_path=args.save_path, force_download=args.force_download, cache_readme=args.cache_readme)