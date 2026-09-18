# Spam SMS Classifier (TF-IDF + Logistic Regression)

A command-line machine learning project that classifies text messages as **spam** or **ham** (not spam) using TF-IDF feature extraction and a Logistic Regression classifier from scikit-learn.

This project was built as a course evaluation project for *Fundamentals of AI and ML*, demonstrating the end-to-end ML pipeline: data preprocessing → feature engineering → model training → evaluation → inference, all runnable from the terminal.

---

## 1. Project Structure

```
spam-classifier/
├── data/
│   └── sample_spam.csv        # small demo dataset (40 labeled messages)
├── models/                    # trained model + vectorizer are saved here
├── outputs/                   # evaluation results (JSON) are saved here
├── src/
│   ├── preprocess.py          # text cleaning + dataset loading
│   ├── train.py                # CLI: trains and evaluates the model
│   ├── predict.py              # CLI: classifies new messages
│   └── convert_uci_dataset.py  # helper: converts the full UCI dataset to the expected format
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

**Requirements:** Python 3.9+ and pip.

1. Clone the repository:
   ```bash
   git clone https://github.com/{github-username}/{repo-name}.git
   cd {repo-name}
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

No API keys, external services, or GPU are required. Everything runs locally and offline once the dataset is present.

---

## 3. Dataset

A small **demo dataset** (`data/sample_spam.csv`, 40 labeled messages) is included so the full pipeline can be run immediately with no setup, for quick testing.

For meaningful evaluation results, it is recommended to train on the full **UCI SMS Spam Collection dataset** (5,574 messages):

1. Download the dataset (search "UCI SMS Spam Collection Dataset" — also mirrored on Kaggle as `spam.csv`).
2. Place the raw file at `data/raw_spam.csv`.
3. Convert it to the expected `label,text` format:
   ```bash
   python src/convert_uci_dataset.py --input data/raw_spam.csv --output data/spam_full.csv
   ```
4. Train using this larger file instead of the sample (see below).

Any dataset works as long as the CSV has a `label` column (`spam`/`ham`) and a `text` column.

---

## 4. How to Run

All commands are run from the **project root**.

### Step 1 — Train the model

Using the included demo dataset:
```bash
python src/train.py --data data/sample_spam.csv --model_dir models --outputs_dir outputs
```

Or, using the full downloaded dataset:
```bash
python src/train.py --data data/spam_full.csv --model_dir models --outputs_dir outputs
```

This will:
- Clean and split the data (80/20 train/test by default)
- Fit a TF-IDF vectorizer and Logistic Regression model
- Print accuracy, precision, recall, F1-score, and a confusion matrix to the terminal
- Save the trained model to `models/model.joblib` and `models/vectorizer.joblib`
- Save a full evaluation report to `outputs/evaluation_results.json`

### Step 2 — Classify a new message

```bash
python src/predict.py --model_dir models --text "Congratulations! You have won a free prize, click here to claim."
```

Or classify multiple messages from a text file (one message per line):
```bash
python src/predict.py --model_dir models --file path/to/messages.txt
```

Example output:
```
Message                                                      Prediction Confidence
-------------------------------------------------------------------------------------
Congratulations! You have won a free prize, click here...    spam       0.87
```

---

## 5. Configuration Options

`train.py` accepts several optional flags:

| Flag             | Default | Description                                  |
|------------------|---------|-----------------------------------------------|
| `--test_size`    | 0.2     | Fraction of data held out for testing         |
| `--random_state` | 42      | Seed for reproducibility                      |
| `--max_features` | 5000    | Max TF-IDF vocabulary size                    |
| `--text_col`     | text    | Name of the text column in the input CSV      |
| `--label_col`    | label   | Name of the label column in the input CSV     |

Run `python src/train.py --help` or `python src/predict.py --help` for the full list.

---

## 6. Notes

- The classifier is intentionally simple (TF-IDF + Logistic Regression) so that every step of the pipeline is transparent and explainable, per the course's fundamentals focus.
- `class_weight="balanced"` is used to handle any class imbalance in the dataset.
- The demo dataset is small (40 rows) purely to make the pipeline runnable out of the box; results on it are not representative of real-world performance — use the full UCI dataset for actual evaluation.
