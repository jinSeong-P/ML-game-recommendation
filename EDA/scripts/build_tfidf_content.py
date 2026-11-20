"""Freeze A-first/C-short descriptions, tokens and a sparse TF-IDF catalog.

No review histories, labels, holdout targets or ranking metrics are loaded.
"""

import argparse
import json
import platform
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy import sparse

from eda_large_review_frank import EDA, digest, log, quote
from tfidf_content import (CLEANING_CONFIG, DEFAULT_INDEX, VECTORIZER_CONFIG,
                           clean_description, has_raw_text, make_vectorizer,
                           script_profile, select_description, sha_text,
                           tokenize_description)


VERSION = "description-tfidf-v1"


def read_sources():
    source_manifest = EDA / "data/source_manifest.json"
    source = json.loads(source_manifest.read_text("utf-8"))
    ref_manifest = EDA / "data/reference/manifest.json"
    reference = json.loads(ref_manifest.read_text("utf-8"))
    baseline = source["modeling_baseline"]["content"]
    if (baseline["primary_field"], baseline["fallback_field"], baseline["tags_and_genres"]) != (
            "A.description", "C.short_description", False):
        raise ValueError("Description source policy differs from confirmed baseline")
    split = EDA / source["split_artifacts"]["directory"]
    split_manifest_path = split / "manifest.json"
    if digest(split_manifest_path) != source["split_artifacts"]["manifest_sha256"]:
        raise ValueError("Frozen split manifest changed")
    split_manifest = json.loads(split_manifest_path.read_text("utf-8"))
    if split_manifest["status"] != "complete" or not split_manifest["validation"]["passed"]:
        raise ValueError("A verified fixed split is required")
    paths = {
        "input_apps": split / "catalog/input_apps.parquet",
        "linux_candidates": split / "catalog/linux_candidates.parquet",
        "A.description": EDA / "legacy/data/processed/A_games_metadata.parquet",
        "C.short_description": EDA / "legacy/data/processed/C_games_2025.parquet",
    }
    old_hashes = {r["file"]: r["sha256"] for r in reference["former_catalog_inputs"]}
    provenance = {"source_manifest_sha256_before_build": digest(source_manifest),
                  "reference_manifest_sha256": digest(ref_manifest),
                  "split_manifest_sha256": digest(split_manifest_path),
                  "sources": {}, "review_data_or_ranking_metrics_loaded": False}
    for name, path in paths.items():
        expected = (old_hashes[path.name] if name in {"A.description", "C.short_description"}
                    else split_manifest["files"]["catalog/" + path.name]["sha256"])
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Pinned source fingerprint changed: {name}")
        provenance["sources"][name] = {"file": path.relative_to(EDA).as_posix(),
                                          "sha256": actual, "bytes": path.stat().st_size}
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        for source_name, id_column in (("A.description", "app_id"), ("C.short_description", "appid")):
            if con.execute(f"SELECT count(*)<>count(DISTINCT {id_column}) FROM read_parquet({quote(paths[source_name])})").fetchone()[0]:
                raise ValueError(f"Duplicate metadata App IDs: {source_name}")
        rows = con.execute(f"""SELECT c.app_id,c.linux,a.description a_description,
            b.short_description c_short_description
            FROM read_parquet({quote(paths['input_apps'])}) c
            LEFT JOIN read_parquet({quote(paths['A.description'])}) a USING(app_id)
            LEFT JOIN read_parquet({quote(paths['C.short_description'])}) b ON c.app_id=b.appid
            ORDER BY c.app_id""").fetchall()
    finally:
        con.close()
    if len(rows) != 37610 or sum(row[1] for row in rows) != 6630:
        raise ValueError("Fixed catalog shape changed")
    return rows, provenance


def prepare_rows(source_rows):
    records = []
    for row_id, (app_id, linux, a_text, c_text) in enumerate(source_rows):
        raw, source = select_description(a_text, c_text)
        clean = clean_description(raw)
        tokens = tokenize_description(clean)
        counts = Counter(tokens)
        status = ("raw_missing" if raw is None else "empty_after_cleaning" if not clean
                  else "zero_after_tokenization" if not tokens else "supported")
        records.append({
            "row_id": row_id, "app_id": app_id, "linux": linux,
            "a_description": a_text, "c_short_description": c_text,
            "text_source": source, "raw_description": raw, "description_clean": clean,
            "raw_sha256": sha_text(raw) if raw is not None else None,
            "clean_sha256": sha_text(clean),
            "raw_char_count": len(raw) if raw is not None else 0,
            "clean_char_count": len(clean), "tokens": tokens, "token_count": len(tokens),
            "unique_token_count": len(counts), "text_status": status,
            "script_profile": script_profile(clean),
            "language_status": "not_automatically_identified",
            "has_replacement_character": "\ufffd" in clean,
            "high_token_repetition": len(tokens) >= 20 and max(counts.values(), default=0) / len(tokens) >= 0.35,
            "cleaning_changed": raw is not None and clean != raw,
            "clean_empty_with_unused_c_alternative": bool(raw is not None and not clean
                                                          and source == "A.description" and has_raw_text(c_text)),
        })
    duplicates = defaultdict(list)
    for record in records:
        if record["description_clean"]:
            duplicates[record["clean_sha256"]].append(record["app_id"])
    groups = []
    for record in records:
        record["duplicate_text_group_size"] = len(duplicates.get(record["clean_sha256"], []))
    for text_hash, apps in sorted(duplicates.items()):
        if len(apps) > 1:
            groups.append({"clean_sha256": text_hash, "app_ids": apps, "group_size": len(apps)})
    return records, groups


def _write_table(records, path, schema=None):
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path, compression="zstd")


def file_inventory(root):
    files = {}
    for path in sorted(root.iterdir()):
        if path.name == "manifest.json":
            continue
        info = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            info["rows"] = pf.metadata.num_rows
            info["schema"] = [{"name": field.name, "type": str(field.type)} for field in pf.schema_arrow]
        files[path.name] = info
    return files


def build(output=DEFAULT_INDEX):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing content artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir()
    manifest = {"version": VERSION, "status": "building",
                "started_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = stage / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    save()
    try:
        log("Verifying pinned metadata sources and fixed input/candidate catalog")
        rows, provenance = read_sources()
        records, duplicates = prepare_rows(rows)
        texts = [row["description_clean"] for row in records]
        vectorizer = make_vectorizer()
        matrix = vectorizer.fit_transform(texts).tocsr()
        matrix.sort_indices()
        feature_names = vectorizer.get_feature_names_out()
        nnz = np.diff(matrix.indptr)
        for record, feature_count in zip(records, nnz):
            record["feature_count"] = int(feature_count)
        log(f"Built TF-IDF: {matrix.shape[0]:,} games x {matrix.shape[1]:,} terms, {matrix.nnz:,} nonzeros")
        _write_table(records, stage / "descriptions.parquet")
        row_fields = ("row_id", "app_id", "linux", "text_source", "text_status", "feature_count")
        _write_table([{field: record[field] for field in row_fields} for record in records],
                     stage / "game_rows.parquet")
        _write_table([{"feature_index": i, "term": str(term), "idf": float(vectorizer.idf_[i])}
                      for i, term in enumerate(feature_names)], stage / "vocabulary_idf.parquet",
                     pa.schema([("feature_index", pa.int32()), ("term", pa.string()), ("idf", pa.float32())]))
        _write_table(duplicates, stage / "duplicate_text_groups.parquet",
                     pa.schema([("clean_sha256", pa.string()), ("app_ids", pa.list_(pa.int64())),
                                ("group_size", pa.int64())]))
        sparse.save_npz(stage / "tfidf_csr.npz", matrix, compressed=True)
        (stage / "parameters.json").write_text(json.dumps({"cleaning": CLEANING_CONFIG,
            "tfidf": VECTORIZER_CONFIG}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        def summarize(subset):
            return {"apps": len(subset), "raw_available": sum(r["raw_description"] is not None for r in subset),
                    "A_primary": sum(r["text_source"] == "A.description" for r in subset),
                    "C_short_fills": sum(r["text_source"] == "C.short_description" for r in subset),
                    "status_counts": dict(Counter(r["text_status"] for r in subset)),
                    "nonzero_vectors": sum(r["feature_count"] > 0 for r in subset),
                    "zero_vectors": sum(r["feature_count"] == 0 for r in subset),
                    "script_profiles": dict(Counter(r["script_profile"] for r in subset)),
                    "replacement_character_apps": sum(r["has_replacement_character"] for r in subset),
                    "high_token_repetition_apps": sum(r["high_token_repetition"] for r in subset),
                    "clean_empty_with_unused_c_alternative": sum(r["clean_empty_with_unused_c_alternative"] for r in subset)}
        manifest.update({"sources": provenance, "cleaning": CLEANING_CONFIG,
            "tfidf": VECTORIZER_CONFIG,
            "fit_scope": "All 37610 fixed input game descriptions including non-Linux and empty rows; shared public metadata only",
            "index": {"format": "scipy CSR float32 NPZ; exact sparse dot product after L2 normalization",
                      "shape": list(matrix.shape), "nnz": int(matrix.nnz),
                      "dedicated_vector_database": False},
            "summary": {"input_catalog": summarize(records),
                        "linux_candidates": summarize([r for r in records if r["linux"]]),
                        "exact_duplicate_groups": len(duplicates),
                        "apps_in_duplicate_groups": sum(g["group_size"] for g in duplicates)},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "pyarrow": pa.__version__},
            "ranking_metrics_computed": False,
            "code_sha256": {"builder": digest(__file__),
                            "text_and_retrieval": digest(Path(__file__).with_name("tfidf_content.py")),
                            "verifier": digest(Path(__file__).with_name("verify_tfidf_content.py"))},
            "files": file_inventory(stage)})
        save()
        from verify_tfidf_content import verify_content
        manifest["verification"] = verify_content(stage)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        stage.rename(output)
        log(f"Published verified content index: {output}")
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"status": manifest["status"], "index": manifest["index"],
                      "summary": manifest["summary"], "verification": manifest["verification"]}, indent=2))


if __name__ == "__main__":
    main()
