# utils.py
# ─────────────────────────────────────────────────────────────────────────────
import re
import pickle
import numpy as np
import spacy
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from scipy.sparse import hstack

# Load spaCy and stopwords once at import
nlp = spacy.load("en_core_web_sm")
STOP_WORDS_SET = set(stopwords.words("english"))
SAFE_DIV = 1e-4

# ──────────── 1. Model / Vectorizer Loaders ─────────────────────────────────
def load_model(model_path="models/model.pkl"):
    """Loads and returns the pickled RandomForest (or XGB) model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_vectorizer(tfidf_path="models/tfidf.pkl"):
    """Loads and returns the pickled TfidfVectorizer."""
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    return tfidf

# ──────────── 2. Text Preprocessing ──────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Lowercases, strips HTML, expands contractions, replaces special symbols,
    lemmatizes with spaCy, and removes punctuation/extra whitespace.
    """
    # 1) Lower, strip, remove HTML
    q = str(text).lower().strip()
    q = BeautifulSoup(q, "html.parser").get_text()

    # 2) Expand contractions
    q = contractions.fix(q)

    # 3) Replace currency/special symbols
    q = (
        q.replace("%", " percent")
         .replace("$", " dollar ")
         .replace("₹", " rupee ")
         .replace("€", " euro ")
         .replace("@", " at ")
         .replace("[math]", "")
    )

    # 4) Convert large-number suffixes
    q = q.replace(",000,000,000 ", "b ")
    q = q.replace(",000,000 ", "m ")
    q = q.replace(",000 ", "k ")
    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)

    # 5) spaCy tokenize → keep lemmas, drop punctuation/whitespace
    doc = nlp(q)
    lemmas = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lemmas.append(token.lemma_)
    return " ".join(lemmas)

# ──────────── 3. Feature‐Engineering for a Single Pair ───────────────────────
def test_common_words(q1: str, q2: str) -> int:
    w1 = set(w.lower().strip() for w in q1.split())
    w2 = set(w.lower().strip() for w in q2.split())
    return len(w1 & w2)

def test_total_words(q1: str, q2: str) -> int:
    w1 = set(w.lower().strip() for w in q1.split())
    w2 = set(w.lower().strip() for w in q2.split())
    return len(w1) + len(w2)

def test_fetch_token_features(q1: str, q2: str) -> list:
    """
    Returns 8 token‐based features for (q1, q2):
     [cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq]
    """
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if not q1_tokens or not q2_tokens:
        return [0.0] * 8

    set1 = set(q1_tokens)
    set2 = set(q2_tokens)
    common_token_count = len(set1 & set2)

    # split into non-stop vs stop
    q1_nonstop = set1 - STOP_WORDS_SET
    q2_nonstop = set2 - STOP_WORDS_SET
    common_word_count = len(q1_nonstop & q2_nonstop)

    q1_stop = set1 & STOP_WORDS_SET
    q2_stop = set2 & STOP_WORDS_SET
    common_stop_count = len(q1_stop & q2_stop)

    min_nonstop = min(len(q1_nonstop), len(q2_nonstop)) + SAFE_DIV
    max_nonstop = max(len(q1_nonstop), len(q2_nonstop)) + SAFE_DIV
    min_stop = min(len(q1_stop), len(q2_stop)) + SAFE_DIV
    max_stop = max(len(q1_stop), len(q2_stop)) + SAFE_DIV

    min_tokens = min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV
    max_tokens = max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV

    return [
        common_word_count / min_nonstop,     # cwc_min
        common_word_count / max_nonstop,     # cwc_max
        common_stop_count / min_stop,        # csc_min
        common_stop_count / max_stop,        # csc_max
        common_token_count / min_tokens,     # ctc_min
        common_token_count / max_tokens,     # ctc_max
        int(q1_tokens[-1] == q2_tokens[-1]), # last_word_eq
        int(q1_tokens[0] == q2_tokens[0]),   # first_word_eq
    ]

import distance  # for longest substring
def test_fetch_length_features(q1: str, q2: str) -> list:
    """
    Returns 3 length‐based features:
     [abs_len_diff, mean_len, longest_substr_ratio]
    """
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if not q1_tokens or not q2_tokens:
        return [0.0] * 3

    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2

    # longest common substring ratio
    substrs = list(distance.lcsubstrings(q1, q2))
    longest = len(substrs[0]) if substrs else 0
    smallest_len = min(len(q1), len(q2)) + 1
    longest_substr_ratio = longest / smallest_len

    return [abs_len_diff, mean_len, longest_substr_ratio]

from fuzzywuzzy import fuzz
def test_fetch_fuzzy_features(q1: str, q2: str) -> list:
    """
    Returns 4 fuzzy‐matching features:
     [fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio]
    """
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]

def query_point_creator(q1: str, q2: str, tfidf=None) -> np.ndarray:
    """
    Given raw q1, q2 strings, returns a 1×(22 + 2*TFIDF_DIM) array:
      [basic_features(7) + token_feats(8) + length_feats(3) + fuzzy_feats(4) ∥ q1_bow ∥ q2_bow].
    Assumes `tfidf` (a fitted TfidfVectorizer) is passed in.
    """
    # 1) Preprocess
    q1p = preprocess(q1)
    q2p = preprocess(q2)

    # 2) Basic features: lengths + word count
    basic = [
        len(q1p),                         # char length q1
        len(q2p),                         # char length q2
        len(q1p.split()),                 # word count q1
        len(q2p.split()),                 # word count q2
        test_common_words(q1p, q2p),      # word_common
        test_total_words(q1p, q2p),       # word_total
        round(test_common_words(q1p, q2p) / (test_total_words(q1p, q2p) + SAFE_DIV), 2),  # word_share
    ]

    # 3) Token features (8 dims)
    tok = test_fetch_token_features(q1p, q2p)

    # 4) Length features (3 dims)
    length = test_fetch_length_features(q1p, q2p)

    # 5) Fuzzy features (4 dims)
    fuzz_feats = test_fetch_fuzzy_features(q1p, q2p)

    # 6) TF-IDF (bag‐of‐words) features for q1, q2
    #    We assume tfidf has been loaded & is already fitted on the full train set
    q1_bow = tfidf.transform([q1p]).toarray().reshape(-1)  # shape (TFIDF_DIM,)
    q2_bow = tfidf.transform([q2p]).toarray().reshape(-1)  # same shape

    # 7) Stack everything horizontally
    return np.hstack((np.array(basic + tok + length + fuzz_feats), q1_bow, q2_bow))
