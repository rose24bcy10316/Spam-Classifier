"""
convert_uci_dataset.py
-----------------------
Converts the raw UCI SMS Spam Collection dataset (commonly distributed
as 'spam.csv' with columns v1,v2,...) into the label,text CSV format
expected by train.py.

Usage:
    python src/convert_uci_dataset.py --input data/raw_spam.csv --output data/spam_full.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw UCI spam.csv file.")
    parser.add_argument("--output", required=True, help="Path to write the cleaned label,text CSV.")
    args = parser.parse_args()

    # The raw UCI file is typically latin-1 encoded with columns v1 (label), v2 (text)
    df = pd.read_csv(args.input, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df = df.dropna()

    df.to_csv(args.output, index=False)
    print(f"Converted {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
