# Generate Tokenizer
from preprocess import tokenize
import argparse
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_words', type=int, default=25000)
    parser.add_argument('--num_deps', type=int, default=2000)
    parser.add_argument('--force_download', action='store_true')
    args = parser.parse_args()
    tokenize.train_tokenizer_main(num_words=args.num_words, num_deps=args.num_deps, force_download=args.force_download)