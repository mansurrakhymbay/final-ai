import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_text

BASE = Path(__file__).resolve().parents[1]
DEPRESSIVE_PATH = BASE / "depressive_tweets_processed.csv"
SENTIMENT_PATH = BASE / "Sentiment Analysis Dataset 2.csv"
OUT_DIR = BASE / "artifacts_4models"
OUT_DIR.mkdir(exist_ok=True)

TEXT_COL = "cleaned_text"
LABEL_COL = "label"

# Limit rows for faster iterations; set to None to use all.
SAMPLE_ROWS = 50000


def load_depressive() -> pd.DataFrame:
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
        DEPRESSIVE_PATH,
        sep="|",
        header=None,
        names=colnames,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    return df[["Text"]].copy()


def load_sentiment() -> pd.DataFrame:
    return pd.read_csv(
        SENTIMENT_PATH,
        usecols=["SentimentText"],
        skipinitialspace=True,
    ).rename(columns={"SentimentText": "Text"})


# Build combined balanced dataset (depressive=1, sampled sentiment=0).
dep_df = load_depressive()
sent_df = load_sentiment()

dep_df[TEXT_COL] = dep_df["Text"].astype(str).apply(preprocess_text)
sent_df[TEXT_COL] = sent_df["Text"].astype(str).apply(preprocess_text)

dep_df = dep_df[dep_df[TEXT_COL].str.len() > 0].copy()
sent_df = sent_df[sent_df[TEXT_COL].str.len() > 0].copy()

dep_df[LABEL_COL] = 1
n_dep = len(dep_df)
n_non = min(len(sent_df), n_dep)
sent_sample = sent_df.sample(n=n_non, random_state=42).copy()
sent_sample[LABEL_COL] = 0

df = pd.concat(
    [
        dep_df[[TEXT_COL, LABEL_COL]],
        sent_sample[[TEXT_COL, LABEL_COL]],
    ],
    ignore_index=True,
)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
    df = df.sample(SAMPLE_ROWS, random_state=42)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True,
)
vectorizer.fit(df[TEXT_COL])

joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")
print("Saved:", OUT_DIR / "tfidf_vectorizer.joblib")
