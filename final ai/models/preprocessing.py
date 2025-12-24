import re
import string
import numpy as np

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

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text
