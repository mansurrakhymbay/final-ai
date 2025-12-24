import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Depression Tweet Classifier",
    layout="centered"
)

import re
import string
from pathlib import Path
import numpy as np
import joblib
import ftfy

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = Path("/Users/Admin/Downloads/final/final/depression_svm_out/svm_depression_model.joblib")
VECT_PATH  = Path("/Users/Admin/Downloads/final/final/depression_svm_out/tfidf_vectorizer.joblib")

# -----------------------
# Preprocessing
# -----------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

def preprocess_text(text):
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = ftfy.fix_text(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -----------------------
# UI
# -----------------------
st.title("Depression Tweet Classifier")
st.caption("SVM (LinearSVC) + TF-IDF")

text = st.text_area(
    "Введите текст:",
    value="i feel empty and hopeless lately",
    height=140
)

if st.button("Classify", type="primary"):
    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])
    score = float(model.decision_function(X)[0])
    confidence = 1 / (1 + np.exp(-abs(score)))

    if pred == 1:
        st.error("Depressive (1)")
    else:
        st.success("Non-depressive (0)")

    st.write(f"Margin: `{score:.4f}`")
    st.write(f"Confidence (pseudo): `{confidence:.3f}`")

    st.divider()
    st.subheader("Cleaned text")
    st.code(cleaned if cleaned else "[empty after cleaning]")
