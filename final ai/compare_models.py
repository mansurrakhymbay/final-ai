import csv
from pathlib import Path
from typing import List, Optional
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "models"))
from preprocessing import preprocess_text  # noqa: E402

ART_DIR = BASE / "artifacts_4models"
SVM_DIR = BASE / "depression_svm_out"

# Choose dataset:
#   - USE_COMBINED = True evaluates on the same combined set (depressive + sampled sentiment) the SVM was trained on.
#   - USE_SENTIMENT = True evaluates on Sentiment Analysis Dataset 2 only.
USE_COMBINED = True
USE_SENTIMENT = False

if USE_COMBINED:
    DEPRESSIVE_PATH = BASE / "depressive_tweets_processed.csv"
    SENTIMENT_PATH = BASE / "Sentiment Analysis Dataset 2.csv"
    TEXT_COL = "cleaned_text"
    LABEL_COL = "label"
elif USE_SENTIMENT:
    DATA_PATH = BASE / "Sentiment Analysis Dataset 2.csv"
    TEXT_COL = "SentimentText"
    LABEL_COL = "Sentiment"
else:
    # Default back to depressive only (not typically used)
    DATA_PATH = BASE / "depressive_tweets_processed.csv"
    TEXT_COL = "text"
    LABEL_COL = "label"

# Reduce runtime if needed; set to None to use all rows.
SAMPLE_ROWS: Optional[int] = 50000


def load_depressive_dataset(path: Path) -> pd.DataFrame:
    colnames = [
        "ID",
        "Date",
        "Time",
        "Timezone",
        "User",
        "Text",
        "Polarity",
        "Col8",
        "Col9",
        "Col10",
    ]
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=colnames,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    return df[["Text"]].copy()


def load_sentiment_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    text_col = None
    for c in df.columns:
        if c.lower() in ["sentimenttext", "text", "sentence", "content"]:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"No text column found in sentiment dataset. Columns: {list(df.columns)}")
    return df[[text_col]].rename(columns={text_col: "Text"}).copy()


def load_data() -> pd.DataFrame:
    if USE_COMBINED:
        dep_df = load_depressive_dataset(DEPRESSIVE_PATH)
        sent_df = load_sentiment_dataset(SENTIMENT_PATH)

        dep_df["cleaned_text"] = dep_df["Text"].astype(str).apply(preprocess_text)
        sent_df["cleaned_text"] = sent_df["Text"].astype(str).apply(preprocess_text)

        dep_df = dep_df[dep_df["cleaned_text"].str.len() > 0].copy()
        sent_df = sent_df[sent_df["cleaned_text"].str.len() > 0].copy()

        dep_df["label"] = 1
        n_dep = len(dep_df)
        n_non = min(len(sent_df), n_dep)
        sent_sample = sent_df.sample(n=n_non, random_state=42).copy()
        sent_sample["label"] = 0

        df = pd.concat(
            [
                dep_df[["cleaned_text", "label"]],
                sent_sample[["cleaned_text", "label"]],
            ],
            ignore_index=True,
        )
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    elif USE_SENTIMENT:
        df = pd.read_csv(
            DATA_PATH,
            usecols=[TEXT_COL, LABEL_COL],
            skipinitialspace=True,
        )
        df[TEXT_COL] = df[TEXT_COL].astype(str).apply(preprocess_text)
        df[LABEL_COL] = df[LABEL_COL].astype(int)
    else:
        cols = [
            "tweet_id",
            "date",
            "time",
            "timezone",
            "username",
            "text",
            "label",
            "retweet_count",
            "favorite_count",
            "extra",
        ]
        df = pd.read_csv(
            DATA_PATH,
            sep="|",
            names=cols,
            header=None,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
        df = df.drop(columns=["extra"])
        df[TEXT_COL] = df[TEXT_COL].astype(str).apply(preprocess_text)
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
        df = df.sample(SAMPLE_ROWS, random_state=42)
    return df


def evaluate(model_path: Path, vect_path: Path, X_text: pd.Series, y_true: pd.Series):
    vectorizer = joblib.load(vect_path)
    model = joblib.load(model_path)
    X = vectorizer.transform(X_text)
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main():
    df = load_data()
    X_text, y_true = df[TEXT_COL], df[LABEL_COL]

    configs: List[dict] = [
        {
            "name": "Naive Bayes",
            "model": ART_DIR / "nb_model.joblib",
            "vectorizer": ART_DIR / "tfidf_vectorizer.joblib",
        },
        {
            "name": "Random Forest",
            "model": ART_DIR / "rf_model.joblib",
            "vectorizer": ART_DIR / "tfidf_vectorizer.joblib",
        },
        {
            "name": "XGBoost",
            "model": ART_DIR / "xgb_model.joblib",
            "vectorizer": ART_DIR / "tfidf_vectorizer.joblib",
        },
        {
            "name": "SVM (depressive+sentiment)",
            "model": SVM_DIR / "svm_depression_model.joblib",
            "vectorizer": SVM_DIR / "tfidf_vectorizer.joblib",
        },
    ]

    results = []
    for cfg in configs:
        if not cfg["model"].exists() or not cfg["vectorizer"].exists():
            results.append(
                {
                    "model": cfg["name"],
                    "status": "missing artifact",
                    "accuracy": None,
                    "f1": None,
                }
            )
            continue

        try:
            metrics = evaluate(cfg["model"], cfg["vectorizer"], X_text, y_true)
            results.append(
                {
                    "model": cfg["name"],
                    "status": "ok",
                    **metrics,
                }
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                {
                    "model": cfg["name"],
                    "status": f"error: {exc}",
                    "accuracy": None,
                    "f1": None,
                }
            )

    summary = pd.DataFrame(results)
    print(f"Rows evaluated: {len(df)} (SAMPLE_ROWS={SAMPLE_ROWS}, USE_COMBINED={USE_COMBINED}, USE_SENTIMENT={USE_SENTIMENT})")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "-"))


if __name__ == "__main__":
    main()
