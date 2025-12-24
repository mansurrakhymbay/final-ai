import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocessing import preprocess_text

BASE = Path(__file__).resolve().parents[1]
DEPRESSIVE_PATH = BASE / "depressive_tweets_processed.csv"
SENTIMENT_PATH = BASE / "Sentiment Analysis Dataset 2.csv"
OUT_DIR = BASE / "artifacts_4models"
VECT_PATH = OUT_DIR / "tfidf_vectorizer.joblib"

TEXT_COL = "cleaned_text"
LABEL_COL = "label"

# Optional downsample for faster iterations.
SAMPLE_ROWS = 50000  # set to None to use all rows


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


def load_combined() -> pd.DataFrame:
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
    return df


df = load_combined()
y = df[LABEL_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = joblib.load(VECT_PATH)
Xtr = vectorizer.transform(X_train)
Xte = vectorizer.transform(X_test)

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=150,
    max_depth=4,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
)
model.fit(Xtr, y_train)

pred = model.predict(Xte)

print("XGB Accuracy:", accuracy_score(y_test, pred))
print("XGB F1:", f1_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))

joblib.dump(model, OUT_DIR / "xgb_model.joblib")
print("Saved:", OUT_DIR / "xgb_model.joblib")
