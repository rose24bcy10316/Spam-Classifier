"""
train.py
--------
Trains a TF-IDF + Logistic Regression spam classifier and saves the
fitted vectorizer + model to disk, along with an evaluation report.

Usage:
    python src/train.py --data data/sample_spam.csv --model_dir models --outputs_dir outputs
"""

import argparse
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a spam SMS/email classifier.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset (text,label columns).")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--max_features", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    print(f"[1/5] Loading and cleaning dataset from {args.data} ...")
    texts, labels = load_dataset(args.data, args.text_col, args.label_col)
    print(f"      Loaded {len(texts)} samples.")

    print("[2/5] Splitting into train/test sets ...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )

    print("[3/5] Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[4/5] Training Logistic Regression classifier ...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    print("[5/5] Evaluating on held-out test set ...")
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
    }
    labels_sorted = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(f"\nConfusion matrix (labels={labels_sorted}):")
    print(cm)
    print("\nFull classification report:")
    print(report)

    # Persist model artifacts
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))

    # Persist evaluation results
    eval_path = os.path.join(args.outputs_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "confusion_matrix": cm,
                "labels_order": labels_sorted,
                "classification_report": report,
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to {args.model_dir}/")
    print(f"Saved evaluation report to {eval_path}")


if __name__ == "__main__":
    main()
