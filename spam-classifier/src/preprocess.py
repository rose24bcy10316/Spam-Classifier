"""
preprocess.py
--------------
Text cleaning and preprocessing utilities for the spam classifier.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Lowercases, removes URLs, numbers, punctuation and extra whitespace
    from a raw message string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                        # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def load_dataset(path: str, text_col: str = "text", label_col: str = "label"):
    """
    Loads a CSV dataset with text and label columns, cleans the text,
    and returns (texts, labels) as parallel lists.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Dataset must contain '{text_col}' and '{label_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 0]

    return df[text_col].tolist(), df[label_col].tolist()
