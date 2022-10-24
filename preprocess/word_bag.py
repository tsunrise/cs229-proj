from typing import List, Tuple
from preprocess.prepare import Crate
import markdown
from bs4 import BeautifulSoup
import tqdm
import numpy as np

def get_words(crate: Crate):
    markdown_text = crate.description + ' \n ' + crate.readme
    # convert markdown to a list of words
    html = markdown.markdown(markdown_text)
    texts = BeautifulSoup(html, 'html.parser').findAll(text=True)
    words_lst = (text.split() for text in texts)
    for words in words_lst:
        for word in words:
            # keep only alphabetic characters
            word = ''.join(c for c in word if c.isalpha())
            if word:
                yield word.lower()

def create_dictionary(crates: List[Crate]):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    ctr = {}
    for crate in tqdm.tqdm(crates, desc='Creating dictionary'):
        for word in set(get_words(crate)):
            ctr.setdefault(word, 0)
            ctr[word] += 1
    
    return {word: i for i, word in enumerate(word for (word, count) in ctr.items() if count >= 3)}

def transform_crate(crates: List[Crate], dictionary: dict, num_categories: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        `X`: A numpy array of shape (num_crates, num_words) containing the word counts for each crate.
        `y`: A numpy array of shape (num_crates, num_categories) containing one-hot encoded categories (can be multiple).
    """

    X = np.zeros((len(crates), len(dictionary)))
    y = np.zeros((len(crates), num_categories))
    for i, msg in tqdm.tqdm(enumerate(crates), desc='Transforming crates', total=len(crates)):
        for word in get_words(msg):
            if word in dictionary:
                X[i, dictionary[word]] += 1
        for category_idx in msg.category_indices:
            y[i, category_idx] = 1

    return X, y