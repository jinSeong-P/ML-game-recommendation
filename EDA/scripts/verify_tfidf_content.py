"""Read back source priority, text/tokens, row mapping and TF-IDF mathematics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest, log
from tfidf_content import (CLEANING_CONFIG, DEFAULT_INDEX, VECTORIZER_CONFIG, TfidfCatalog,
                           clean_description, select_description, sha_text,
                           tokenize_description)


def verify_content(folder=DEFAULT_INDEX):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    required = {"descriptions.parquet", "game_rows.parquet", "vocabulary_idf.parquet",
                "duplicate_text_groups.parquet", "tfidf_csr.npz", "parameters.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete content artifact inventory")
    for relative, info in manifest["files"].items():
        path = folder / relative
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Content artifact fingerprint changed: {relative}")
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != info["rows"]:
            raise ValueError(f"Content artifact row count changed: {relative}")
    parameters = json.loads((folder / "parameters.json").read_text("utf-8"))
    if parameters != {"cleaning": CLEANING_CONFIG, "tfidf": VECTORIZER_CONFIG}:
        raise ValueError("Cleaning/TF-IDF parameters differ from versioned code")
    records = pq.read_table(folder / "descriptions.parquet").to_pylist()
    rows = pq.read_table(folder / "game_rows.parquet").to_pydict()
    terms = pq.read_table(folder / "vocabulary_idf.parquet").to_pydict()
    matrix = sparse.load_npz(folder / "tfidf_csr.npz").tocsr()
    checks = {}
    def require(name, violations):
        checks[name] = int(violations)
        if violations:
            raise ValueError(f"Content verification failed: {name}={violations}")
    require("row_mapping_errors", rows["row_id"] != list(range(len(records))) or
            [r["app_id"] for r in records] != rows["app_id"] or
            [r["linux"] for r in records] != rows["linux"] or
            rows["app_id"] != sorted(set(rows["app_id"])))
    source_input = EDA / manifest["sources"]["sources"]["input_apps"]["file"]
    source_linux = EDA / manifest["sources"]["sources"]["linux_candidates"]["file"]
    for name, path in (("input_apps", source_input), ("linux_candidates", source_linux)):
        if digest(path) != manifest["sources"]["sources"][name]["sha256"]:
            raise ValueError(f"Frozen source catalog changed: {name}")
    input_rows = pq.read_table(source_input).to_pydict()
    linux_rows = pq.read_table(source_linux).to_pydict()
    require("fixed_catalog_differences", rows["app_id"] != input_rows["app_id"] or rows["linux"] != input_rows["linux"])
    require("linux_candidate_differences", [app for app, linux in zip(rows["app_id"], rows["linux"]) if linux] != linux_rows["app_id"])
    require("source_priority_errors", sum((r["raw_description"], r["text_source"]) !=
            select_description(r["a_description"], r["c_short_description"]) for r in records))
    require("cleaning_errors", sum(r["description_clean"] != clean_description(r["raw_description"]) for r in records))
    require("tokenization_errors", sum(r["tokens"] != tokenize_description(r["description_clean"]) or
            r["token_count"] != len(r["tokens"]) for r in records))
    require("text_hash_errors", sum(r["clean_sha256"] != sha_text(r["description_clean"]) or
            (r["raw_sha256"] != sha_text(r["raw_description"]) if r["raw_description"] is not None
             else r["raw_sha256"] is not None) for r in records))
    require("matrix_shape_errors", matrix.shape != (len(records), len(terms["term"])) or
            matrix.shape != tuple(manifest["index"]["shape"]) or matrix.nnz != manifest["index"]["nnz"])
    require("invalid_matrix_values", matrix.dtype != np.float32 or
            not np.all(np.isfinite(matrix.data)) or np.any(matrix.data <= 0))
    require("vocabulary_index_errors", terms["feature_index"] != list(range(matrix.shape[1])) or
            terms["term"] != sorted(set(terms["term"])))
    actual_nnz = np.diff(matrix.indptr)
    require("feature_status_errors", sum(int(n) != r["feature_count"] or
            (n > 0) != (r["text_status"] == "supported") for n, r in zip(actual_nnz, records)))
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    require("l2_norm_errors", np.count_nonzero(np.abs(norms[actual_nnz > 0] - 1) > 1e-5))
    # Independently reconstruct term frequencies and the smoothed IDF formula
    # from saved cleaned documents, rather than trusting the fitted estimator.
    count = CountVectorizer(vocabulary={term: i for i, term in enumerate(terms["term"])},
                            tokenizer=tokenize_description, token_pattern=None,
                            lowercase=False, dtype=np.float32)
    frequencies = count.transform([r["description_clean"] for r in records]).tocsr()
    document_frequency = np.asarray((frequencies > 0).sum(axis=0)).ravel()
    expected_idf = (np.log((1 + matrix.shape[0]) / (1 + document_frequency)) + 1).astype(np.float32)
    stored_idf = np.array(terms["idf"], dtype=np.float32)
    require("idf_formula_errors", np.count_nonzero(np.abs(expected_idf - stored_idf) > 2e-6))
    expected_matrix = normalize(frequencies.multiply(stored_idf).tocsr(), norm="l2").tocsr()
    delta = expected_matrix - matrix
    require("tfidf_reconstruction_errors", np.count_nonzero(np.abs(delta.data) > 2e-6))
    require("token_vocabulary_gaps", np.count_nonzero(document_frequency == 0) or
            len({token for r in records for token in r["tokens"]}) != len(terms["term"]))
    index = TfidfCatalog(rows["app_id"], rows["linux"], matrix)
    smoke = next((int(app) for app, n in zip(rows["app_id"], actual_nnz) if n > 0), None)
    result = index.recommend([smoke]) if smoke is not None else []
    require("retrieval_smoke_errors", len(result) > 10 or any(r["app_id"] == smoke or
            r["cosine"] <= 0 or not index.linux[index.row_by_app[r["app_id"]]] for r in result))
    require("zero_profile_fallback_errors", len(index.recommend([])))
    log("Read-back content verification: source policy, tokens, IDF, L2 vectors and cosine retrieval passed")
    return {"passed": True, "hashes_verified": True, "zero_violation_checks": checks,
            "input_apps": len(records), "linux_candidates": sum(rows["linux"]),
            "vocabulary_size": len(terms["term"]), "zero_vectors": int(np.count_nonzero(actual_nnz == 0)),
            "ranking_metrics_computed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    print(json.dumps(verify_content(args.folder), indent=2))


if __name__ == "__main__":
    main()
