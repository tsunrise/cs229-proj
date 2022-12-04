# Generate Tokenizer
from preprocess import tokenize
import argparse
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_words', type=int, default=25000, help='Number of words to keep in the vocabulary')
    parser.add_argument('--num_feats', type=int, default=2500, help="Number of feature words")
    parser.add_argument('--force_download', action='store_true')
    args = parser.parse_args()
    tokenize.train_tokenizer_main(num_words=args.num_words, num_feats=args.num_feats, force_download=args.force_download)