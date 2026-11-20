"""Description-only TF-IDF text rules and exact sparse cosine retrieval.

The index uses public fixed-catalog game metadata, never review labels. No
pickle, embedding service, dense all-pairs matrix or vector database is needed.
"""

import argparse
import hashlib
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest


DEFAULT_INDEX = EDA / "data/processed/modeling/tfidf_content_v1"
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)
BB_IMAGE_RE = re.compile(r"\[img(?:=[^\]]*)?\].*?\[/img\]", re.I | re.S)
BB_TAG_RE = re.compile(r"\[/?(?:b|i|u|s|h[1-6]|p|list|olist|\*|url|quote|code|spoiler|table|tr|td)(?:=[^\]]*)?\]", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.I)
BLOCK_TAGS = {"br", "p", "div", "li", "ul", "ol", "hr", "table", "tr", "td",
              "th", "section", "article", "blockquote", "pre", "h1", "h2", "h3",
              "h4", "h5", "h6"}

VECTORIZER_CONFIG = {
    "analyzer": "word", "token_pattern": None,
    "tokenizer": "tfidf_content.tokenize_description", "tokenizer_pattern": TOKEN_PATTERN,
    "lowercase": False, "tokenizer_lowercase": True,
    "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0,
    "max_features": None, "stop_words": None, "norm": "l2", "use_idf": True,
    "smooth_idf": True, "sublinear_tf": False, "binary": False, "dtype": "float32",
    "tuning_status": "untuned_initial_baseline; no validation/test ranking metrics used",
}
CLEANING_CONFIG = {
    "version": "description-clean-v1",
    "source_priority": ["A.description", "C.short_description"],
    "fallback_only_when_primary": "null, empty or Unicode whitespace-only before cleaning",
    "unicode_normalization": "NFKC", "html_entity_decode_passes": 2,
    "html": "strip tags/comments and script/style content; separate block elements",
    "bbcode": "remove known formatting tags and image payloads; preserve URL labels",
    "urls": "remove http/https/www URLs", "control_characters": "replace Cc/Cf with space",
    "whitespace": "collapse to single spaces and trim",
    "promotional_boilerplate": "flag repetition; no semantic sentence deletion without evidence",
    "stemming_or_morphology": False,
    "language_policy": "retain all scripts; script profile is not language identification",
}


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden.append(tag)
        elif not self.hidden and tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif not self.hidden and tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_startendtag(self, tag, attrs):
        if not self.hidden and tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def has_raw_text(value):
    return isinstance(value, str) and bool(value.strip())


def select_description(a_description, c_short_description):
    if has_raw_text(a_description):
        return a_description, "A.description"
    if has_raw_text(c_short_description):
        return c_short_description, "C.short_description"
    return None, "missing"


def clean_description(value):
    if not has_raw_text(value):
        return ""
    text = unicodedata.normalize("NFKC", value)
    for _ in range(2):
        text = html.unescape(text)
    text = BB_IMAGE_RE.sub(" ", text)
    text = BB_TAG_RE.sub(" ", text)
    parser = _PlainText()
    parser.feed(text)
    parser.close()
    text = "".join(parser.parts)
    text = URL_RE.sub(" ", text)
    text = "".join(" " if unicodedata.category(ch) in {"Cc", "Cf"} else ch for ch in text)
    return " ".join(text.split())


def tokenize_description(clean_text):
    return TOKEN_RE.findall(clean_text.lower())


def script_profile(text):
    """Observed writing systems only; this does not infer text language."""
    scripts = set()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        script = next((s for s in ("LATIN", "CYRILLIC", "GREEK", "HIRAGANA",
                                  "KATAKANA", "HANGUL", "ARABIC", "HEBREW", "THAI")
                       if s in name), "CJK" if "CJK" in name else "OTHER")
        scripts.add(script)
    return "+".join(sorted(scripts)) or "NO_LETTERS"


def make_vectorizer():
    return TfidfVectorizer(
        tokenizer=tokenize_description, token_pattern=None, lowercase=False,
        ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=None,
        stop_words=None, norm="l2", use_idf=True, smooth_idf=True,
        sublinear_tf=False, binary=False, dtype=np.float32,
    )


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TfidfCatalog:
    def __init__(self, app_ids, linux, matrix):
        self.app_ids = np.asarray(app_ids, dtype=np.int64)
        self.linux = np.asarray(linux, dtype=bool)
        self.matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        if self.matrix.shape[0] != len(self.app_ids) or len(self.linux) != len(self.app_ids):
            raise ValueError("Index row mapping shape mismatch")
        if len(set(self.app_ids)) != len(self.app_ids) or np.any(self.app_ids <= 0):
            raise ValueError("App IDs must be unique positive integers")
        self.row_by_app = {int(app): i for i, app in enumerate(self.app_ids)}

    @classmethod
    def load(cls, folder=DEFAULT_INDEX, verify_hashes=True):
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        if manifest["status"] != "complete":
            raise ValueError("Refusing incomplete TF-IDF artifact")
        if verify_hashes:
            for relative, info in manifest["files"].items():
                if digest(folder / relative) != info["sha256"]:
                    raise ValueError(f"TF-IDF artifact fingerprint changed: {relative}")
        rows = pq.read_table(folder / "game_rows.parquet").to_pydict()
        if rows["row_id"] != list(range(len(rows["app_id"]))):
            raise ValueError("Game row mapping is not contiguous")
        matrix = sparse.load_npz(folder / "tfidf_csr.npz")
        terms = pq.read_table(folder / "vocabulary_idf.parquet")
        if matrix.shape != (len(rows["app_id"]), terms.num_rows):
            raise ValueError("Vocabulary or game row count differs from matrix")
        return cls(rows["app_id"], rows["linux"], matrix)

    def recommend(self, positive_app_ids, exclude_app_ids=(), limit=10):
        """Mean L2-normalized positive game vectors -> candidate cosine.

        All remaining observed games must be supplied as exclusions by the
        evaluator. Positive inputs are also excluded defensively. No filling.
        """
        if not 0 <= limit <= 10:
            raise ValueError("Baseline returns at most 10 recommendations")
        positive_app_ids = [int(app) for app in positive_app_ids]
        if len(set(positive_app_ids)) != len(positive_app_ids):
            raise ValueError("Positive query App IDs must be unique")
        unknown = set(positive_app_ids) - self.row_by_app.keys()
        if unknown:
            raise ValueError("Positive query contains an App ID outside the fixed catalog")
        if not positive_app_ids or limit == 0:
            return []
        rows = [self.row_by_app[app] for app in positive_app_ids]
        weights = sparse.csr_matrix(np.full((1, len(rows)), 1.0 / len(rows), dtype=np.float32))
        profile = normalize(weights @ self.matrix[rows], norm="l2", copy=False)
        if not profile.nnz:
            return []
        candidates = np.flatnonzero(self.linux)
        scores = (self.matrix[candidates] @ profile.T).toarray().ravel()
        excluded = set(int(app) for app in exclude_app_ids) | set(positive_app_ids)
        valid = (scores > 0) & np.array([int(self.app_ids[r]) not in excluded for r in candidates])
        candidates, scores = candidates[valid], scores[valid]
        order = np.lexsort((self.app_ids[candidates], -scores))[:limit]
        return [{"app_id": int(self.app_ids[candidates[i]]),
                 "cosine": float(np.clip(scores[i], 0.0, 1.0))} for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--positive-apps", nargs="+", type=int, required=True)
    parser.add_argument("--exclude-apps", nargs="*", type=int, default=[])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = TfidfCatalog.load(args.index).recommend(args.positive_apps, args.exclude_apps, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
