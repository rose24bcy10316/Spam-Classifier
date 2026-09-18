"""
predict.py
----------
Loads the trained model + vectorizer and classifies new message(s) as
spam or ham, from the command line.

Usage:
    python src/predict.py --model_dir models --text "You have won a free prize, click here!"
    python src/predict.py --model_dir models --file path/to/messages.txt
"""

import argparse
import os

import joblib

from preprocess import clean_text


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("--model_dir", type=str, default="models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A single message to classify.")
    group.add_argument("--file", type=str, help="Path to a .txt file, one message per line.")
    return parser.parse_args()


def load_artifacts(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            f"Model artifacts not found in '{model_dir}'. Run train.py first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_messages(messages, model, vectorizer):
    cleaned = [clean_text(m) for m in messages]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        confidence = prob[classes.index(pred)]
        results.append((msg, pred, round(float(confidence), 4)))
    return results


def main():
    args = parse_args()
    model, vectorizer = load_artifacts(args.model_dir)

    if args.text:
        messages = [args.text]
    else:
        with open(args.file, "r") as f:
            messages = [line.strip() for line in f if line.strip()]

    results = predict_messages(messages, model, vectorizer)

    print(f"\n{'Message':<60} {'Prediction':<10} {'Confidence'}")
    print("-" * 85)
    for msg, pred, conf in results:
        display_msg = (msg[:57] + "...") if len(msg) > 60 else msg
        print(f"{display_msg:<60} {pred:<10} {conf}")


if __name__ == "__main__":
    main()
