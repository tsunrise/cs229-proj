from typing import List


def markdown_to_text(markdown: str) -> str:
    """Converts a markdown string to a text string."""
    ...

def batch_markdown_to_text(markdowns: List[str]) -> List[str]:
    """Converts a list of markdown strings to a list of text strings."""
    ...

def normalize_text_simple(text: str) -> str:
    """Normalizes a text string by collecting only words, numbers, and punctuations."""
    ...

def batch_normalize_text_simple(texts: List[str]) -> List[str]:
    """Normalizes a list of text strings by collecting only words, numbers, and punctuations."""
    ...