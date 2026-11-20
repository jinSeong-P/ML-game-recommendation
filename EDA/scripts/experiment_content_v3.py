"""Build v3 content model artifacts without reading validation or test outcomes.

The artifacts share the frozen v1 game row order and expose a TF-IDF CSR
matrix, a metadata CSR matrix, or dense sentence embeddings. This builder does
not evaluate recommendations or compute ranking metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from eda_large_review_frank import EDA, digest, log, quote
from improvement_tfidf import normalize_concept, parse_metadata_labels
from tfidf_content import (
    DEFAULT_INDEX,
    clean_description,
    has_raw_text,
    make_vectorizer,
    tokenize_description,
)


VERSION = "content-models-v3"
DEFAULT_OUTPUT = EDA / "data/processed/modeling/improvement_v3/content"
DEFAULT_PROTOCOL = EDA / "data/reference/improvement_protocol_v3.json"
DEFAULT_SPLIT = EDA / "data/processed/modeling/split_v1"
DEFAULT_V1 = DEFAULT_INDEX
DEFAULT_V2_METADATA = EDA / "data/processed/modeling/improvement_v2/tfidf/metadata_channel"
V2_METADATA_REBUILT = EDA / "data/processed/modeling/improvement_v2/tfidf_rebuild_final2/metadata_channel"
DEFAULT_SOURCE_MANIFEST = EDA / "data/source_manifest.json"
DEFAULT_KIERAN_ARCHIVE = EDA / "data/raw/kieran_original/steam-reviews-v2.zip"
DEFAULT_KIERAN_INGESTION = EDA / "data/processed/kieran_original/ingestion_manifest.json"
DEFAULT_FRANK_GAMES = DEFAULT_OUTPUT / "sources/frank_games.parquet"

FRANK_DATASET = "frankjc2022/steam-dataset"
FRANK_COMMIT = "75694cc07a6cdc69f48f9c316ca1e3e38a3f0f89"
FRANK_GAMES_PATH = "data/v1/games.parquet"
FRANK_GAMES_URL = (
    f"https://huggingface.co/datasets/{FRANK_DATASET}/resolve/{FRANK_COMMIT}/"
    f"{FRANK_GAMES_PATH}"
)
FRANK_GAMES_BYTES = 131_986_254
FRANK_GAMES_SHA256 = "455c4e33f91d94909b0f5d03583d59ad2d453ca97cb12c21db4f542eccd003bc"

MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MINILM_MAX_LENGTH = 128
MINILM_BATCH_SIZE = 32

# The fixed ten-app improvement target is the Linux set with neither a usable
# v1 description nor the v2 tag/genre channel. It is set before v3 metrics.
MISSING_DESCRIPTION_TARGET_APPS = (
    400, 550, 620, 8930, 34330, 774361, 1066780, 1066890, 1092790, 1703340,
)
REVIEW_SAMPLE_CAP = 100
REVIEW_SAMPLE_POLICY = "first 100 accepted train review IDs per app in source CSV order"
KIERAN_CSV_BLOCK_BYTES = 128 * 1024 * 1024
STEAM_APPDETAILS_MAX_BYTES = 2_000_000
STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={app_id}&l=english"
STEAM_BASE = 76561197960265728

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "description_missing_fill": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32",
        "description_policy": "Frozen v1 A-first short text; fill the fixed ten unsupported Linux apps from pinned frank games.parquet, then Steam appdetails only when the pinned file has no description.",
        "vectorizer": "frozen_v1_word_unigram_tfidf",
    },
    "description_missing_fill_bigram": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32",
        "description_policy": "Frozen v1 A-first short text; fill the fixed ten unsupported Linux apps from pinned frank games.parquet, then Steam appdetails only when the pinned file has no description.",
        "vectorizer": "v2_T2_word_unigram_bigram_tfidf",
        "comparison": "Same source fill and row mapping as description_missing_fill; unigram+bigram settings match v2 T2.",
    },
    "description_long_word_tfidf": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32",
        "description_policy": "C.about_the_game, then C.detailed_description, then frozen v1 A-first short description.",
        "vectorizer": "word_tfidf_unigram_bigram",
    },
    "metadata_extended_tfidf": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32_components",
        "description_policy": "Frozen v1 description TF-IDF unigram+bigram component plus separate extended metadata TF-IDF component.",
        "metadata_fields": ["v2 tags", "v2 genres", "C categories", "C developers", "C publishers"],
        "score_spec": "0.5 * cosine(description profile, candidate description) + 0.5 * cosine(metadata profile, candidate metadata)",
    },
    "description_char_wb_tfidf": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32",
        "description_policy": "Frozen v1 cleaned descriptions; multilingual character n-grams.",
        "vectorizer": "char_wb_3_5_tfidf",
    },
    "description_sentence_embedding": {
        "family": "content",
        "matrix_kind": "dense_float32_sentence_embedding",
        "description_policy": "Frozen v1 cleaned descriptions; pretrained multilingual sentence embeddings.",
        "model_id": MINILM_MODEL,
        "model_revision": MINILM_REVISION,
    },
    "train_review_body_tfidf": {
        "family": "content",
        "matrix_kind": "scipy_csr_float32",
        "description_policy": "Aggregated review body text from accepted train review IDs only; deterministic per-app cap.",
        "vectorizer": "word_tfidf_unigram_bigram",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def file_inventory(folder: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    excluded = {folder / "manifest.json", folder / "registry.json"}
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p not in excluded):
        if any(".building-" in parent.name for parent in path.parents if parent != folder):
            continue
        info: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            info["rows"] = int(parquet.metadata.num_rows)
            info["schema"] = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
        result[path.relative_to(folder).as_posix()] = info
    return result


def _verify_inventory(folder: Path, inventory: dict[str, dict[str, Any]]) -> None:
    for relative, expected in inventory.items():
        path = folder / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Content artifact fingerprint mismatch: {relative}")
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
            raise ValueError(f"Content artifact row count mismatch: {relative}")


def _require_child(parent: Path, path: Path) -> Path:
    """Resolve a new artifact path and require it to stay under its owned root."""
    root = Path(parent).resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"Refusing a symlink in the content artifact path: {candidate}")
    resolved = candidate.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Content artifact path escapes its output root: {resolved}")
    return resolved


def make_word_vectorizer(ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Use the frozen v1 word tokenizer and standard L2 TF-IDF settings."""
    return TfidfVectorizer(
        analyzer="word", tokenizer=tokenize_description, token_pattern=None,
        lowercase=False, ngram_range=ngram_range, min_df=1, max_df=1.0,
        max_features=None, stop_words=None, norm="l2", use_idf=True,
        smooth_idf=True, sublinear_tf=False, binary=False, dtype=np.float32,
    )


def make_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=1.0,
        max_features=100_000, stop_words=None, norm="l2", use_idf=True,
        smooth_idf=True, sublinear_tf=True, binary=False, dtype=np.float32,
    )


def make_metadata_vectorizer() -> TfidfVectorizer:
    # Each normalized label is already a single token; this retains full
    # phrases and the `tag::`, `genre::`, `category::`, `developer::`, and
    # `publisher::` prefixes.
    return TfidfVectorizer(
        analyzer=lambda labels: labels, min_df=1, max_df=1.0,
        max_features=None, norm="l2", use_idf=True, smooth_idf=True,
        sublinear_tf=False, binary=True, dtype=np.float32,
    )


def _write_vocabulary(path: Path, vectorizer: TfidfVectorizer) -> None:
    terms = vectorizer.get_feature_names_out()
    idf = np.asarray(vectorizer.idf_, dtype=np.float32)
    table = pa.table({
        "feature_index": pa.array(np.arange(len(terms), dtype=np.int32)),
        "term": pa.array([str(term) for term in terms], type=pa.large_string()),
        "idf": pa.array(idf, type=pa.float32()),
    })
    pq.write_table(table, path, compression="zstd")


def _write_source_audit(path: Path, records: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def _save_sparse_matrix(folder: Path, matrix: sparse.spmatrix, name: str = "tfidf_csr.npz") -> dict[str, Any]:
    csr = sparse.csr_matrix(matrix, dtype=np.float32)
    csr.sort_indices()
    if not np.all(np.isfinite(csr.data)):
        raise ValueError("Sparse content matrix contains nonfinite values")
    sparse.save_npz(folder / name, csr, compressed=True)
    return {"path": name, "shape": list(csr.shape), "nnz": int(csr.nnz),
            "dtype": str(csr.dtype), "format": "scipy CSR NPZ"}


def _copy_game_rows(folder: Path, v1_dir: Path = DEFAULT_V1) -> str:
    source = v1_dir / "game_rows.parquet"
    destination = folder / "game_rows.parquet"
    shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise ValueError("Copied game row mapping differs from frozen v1")
    return digest(destination)


def _model_manifest_base(
    model_id: str, output: Path, protocol_path: Path, split_path: Path, v1_dir: Path,
) -> dict[str, Any]:
    source_manifest = DEFAULT_SOURCE_MANIFEST
    return {
        "version": VERSION,
        "model_id": model_id,
        "family": "content",
        "status": "complete",
        "created_utc": utc_now(),
        "parameters": MODEL_CONFIGS[model_id],
        "lineage": {
            "protocol": {"path": protocol_path.relative_to(EDA).as_posix(), "sha256": digest(protocol_path)},
            "split_manifest": {"path": (split_path / "manifest.json").relative_to(EDA).as_posix(),
                               "sha256": digest(split_path / "manifest.json")},
            "v1_content_manifest": {"path": (v1_dir / "manifest.json").relative_to(EDA).as_posix(),
                                    "sha256": digest(v1_dir / "manifest.json")},
            "source_manifest": {"path": source_manifest.relative_to(EDA).as_posix(), "sha256": digest(source_manifest)},
            "fit_scope": "Frozen 37,610 game input catalog including non-Linux games; public game text and metadata, except the review body model which uses accepted train review IDs only.",
            "validation_or_test_labels_loaded_for_fitting": False,
            "frank_review_outcomes_loaded_for_fitting": False,
        },
        "source_lineage": {},
        "mapping": {"path": "game_rows.parquet", "source": "byte-identical copy of frozen v1 game_rows.parquet",
                    "row_order": "v1 row_id 0..37609; all models have exactly the same rows"},
        "ranking_metrics_computed": False,
        "files": {},
    }


def _start_model(folder: Path, model_id: str, v1_dir: Path, protocol_path: Path, split_path: Path) -> tuple[Path, dict[str, Any]]:
    folder = Path(folder)
    if folder.exists():
        manifest_path = folder / "manifest.json"
        if manifest_path.is_file():
            old = read_json(manifest_path)
            if old.get("status") == "complete":
                _verify_inventory(folder, old.get("files", {}))
                if old.get("model_id") != model_id or old.get("protocol_sha256") != digest(protocol_path):
                    raise ValueError(f"Existing artifact does not match current v3 protocol: {folder}")
                return folder, old
        raise FileExistsError(f"Refusing to overwrite incomplete content artifact: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    resumable_stages = sorted(folder.parent.glob(f".{folder.name}.building-*"))
    if resumable_stages:
        if len(resumable_stages) != 1:
            raise FileExistsError(f"Multiple incomplete content stages need manual review: {resumable_stages}")
        stage = resumable_stages[0]
        _require_child(folder.parent, stage)
        snapshot = stage / "builder_snapshot.py"
        mapping = stage / "game_rows.parquet"
        cached_review_keys = stage / ".review_work/accepted_train_review_keys.npy"
        if model_id == "train_review_body_tfidf" and cached_review_keys.is_file():
            if mapping.is_file() and digest(mapping) != digest(v1_dir / "game_rows.parquet"):
                raise FileExistsError(f"Incomplete content stage has a mismatched row mapping: {stage}")
            if not mapping.is_file():
                _copy_game_rows(stage, v1_dir)
            log(f"Resuming review-body stage with accepted-key cache: {model_id}")
            manifest = _model_manifest_base(model_id, folder.parent, protocol_path, split_path, v1_dir)
            manifest["protocol_sha256"] = digest(protocol_path)
            return stage, manifest
        if not snapshot.is_file() or not mapping.is_file() or digest(mapping) != digest(v1_dir / "game_rows.parquet"):
            if any(path.is_file() for path in stage.rglob("*")):
                raise FileExistsError(f"Incomplete content stage cannot be safely resumed: {stage}")
            shutil.rmtree(_require_child(folder.parent, stage), ignore_errors=False)
            log(f"Removed empty abandoned content stage: {stage}")
        else:
            log(f"Resuming incomplete v3 content stage: {model_id}")
            manifest = _model_manifest_base(model_id, folder.parent, protocol_path, split_path, v1_dir)
            manifest["protocol_sha256"] = digest(protocol_path)
            return stage, manifest
    stage = folder.parent / f".{folder.name}.building-{uuid.uuid4().hex[:10]}"
    _require_child(folder.parent, folder)
    _require_child(folder.parent, stage)
    stage.mkdir()
    _copy_game_rows(stage, v1_dir)
    manifest = _model_manifest_base(model_id, folder.parent, protocol_path, split_path, v1_dir)
    manifest["protocol_sha256"] = digest(protocol_path)
    return stage, manifest


def _publish_model(stage: Path, folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _require_child(folder.parent, stage)
    _require_child(folder.parent, folder)
    manifest["files"] = file_inventory(stage)
    write_json(stage / "manifest.json", manifest)
    verify_model_artifact(stage, expected_rows=37_610)
    stage.rename(folder)
    return read_json(folder / "manifest.json")


def verify_model_artifact(folder: Path, expected_rows: int | None = None) -> dict[str, Any]:
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("family") != "content":
        raise ValueError(f"Incomplete v3 content artifact: {folder}")
    _verify_inventory(folder, manifest.get("files", {}))
    rows = pq.read_table(folder / "game_rows.parquet", columns=["row_id", "app_id", "linux"])
    if rows.num_rows != expected_rows if expected_rows is not None else False:
        raise ValueError("Content model mapping has the wrong number of rows")
    row_ids = np.asarray(rows.column("row_id").to_numpy(), dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(len(row_ids), dtype=np.int64)):
        raise ValueError("Content model mapping row_id is not contiguous")
    matrix_info = manifest.get("matrix", {})
    matrix_path = folder / matrix_info.get("path", "")
    if matrix_path.suffix == ".npz":
        matrix = sparse.load_npz(matrix_path)
        shape = matrix.shape
    elif matrix_path.suffix == ".npy":
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        shape = matrix.shape
    else:
        raise ValueError("Manifest matrix path must be .npz or .npy")
    if shape[0] != rows.num_rows or list(shape) != matrix_info.get("shape"):
        raise ValueError("Content matrix rows do not match game_rows mapping")
    for component in manifest.get("components", []):
        path = folder / component["path"]
        data = sparse.load_npz(path) if path.suffix == ".npz" else np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape[0] != rows.num_rows or list(data.shape) != component["shape"]:
            raise ValueError(f"Content matrix component rows do not match game_rows mapping: {path.name}")
    return {"status": "passed", "rows": int(rows.num_rows), "matrix_shape": list(shape),
            "files_verified": len(manifest.get("files", {}))}


def _validate_sources(
    protocol_path: Path = DEFAULT_PROTOCOL, split_path: Path = DEFAULT_SPLIT, v1_dir: Path = DEFAULT_V1,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    split_manifest_path = split_path / "manifest.json"
    v1_manifest_path = v1_dir / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    split_manifest = read_json(split_manifest_path)
    v1_manifest = read_json(v1_manifest_path)
    if protocol.get("status") != "fixed_before_v3_metrics":
        raise ValueError("v3 experiment protocol is not frozen")
    if digest(split_manifest_path) != protocol.get("split_manifest_sha256"):
        raise ValueError("split_v1 manifest differs from the frozen v3 protocol")
    if split_manifest.get("status") != "complete" or not split_manifest.get("validation", {}).get("passed"):
        raise ValueError("split_v1 is not verified")
    if v1_manifest.get("status") != "complete":
        raise ValueError("Frozen v1 content artifact is not complete")
    for relative, info in v1_manifest.get("files", {}).items():
        path = v1_dir / relative
        if not path.is_file() or digest(path) != info["sha256"]:
            raise ValueError(f"Frozen v1 content file changed: {relative}")
    for source_name, info in v1_manifest.get("sources", {}).get("sources", {}).items():
        source_path = EDA / info["file"]
        if not source_path.is_file() or digest(source_path) != info["sha256"]:
            raise ValueError(f"Frozen v1 metadata input changed: {source_name}")
    if protocol.get("content", {}).get("character_tfidf", {}).get("max_features") != 100000:
        raise ValueError("Frozen v3 character TF-IDF settings differ from this builder")
    if protocol.get("content", {}).get("long_description") != "Prefer C about_the_game, then detailed_description, then frozen A-first short description; content snapshot versions unchanged.":
        raise ValueError("Frozen v3 long description policy differs from this builder")
    if protocol.get("train_only") is not True:
        raise ValueError("v3 protocol does not require train-only fitting")
    if source_manifest.get("datasets", {}).get("kieran", {}).get("archive_sha256") is None:
        raise ValueError("Kieran source archive SHA256 missing from source manifest")
    return {
        "protocol_sha256": digest(protocol_path),
        "split_manifest_sha256": digest(split_manifest_path),
        "v1_manifest_sha256": digest(v1_manifest_path),
        "source_manifest_sha256": digest(source_manifest_path),
        "split_manifest": split_manifest,
        "v1_manifest": v1_manifest,
        "source_manifest": source_manifest,
    }


def _load_v1_rows(v1_dir: Path = DEFAULT_V1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    game_rows = pq.read_table(v1_dir / "game_rows.parquet").to_pylist()
    descriptions = pq.read_table(v1_dir / "descriptions.parquet",
                                 columns=["app_id", "raw_description", "description_clean", "text_status"]).to_pylist()
    if len(game_rows) != 37_610 or len(descriptions) != len(game_rows):
        raise ValueError("Frozen v1 catalog row count changed")
    if [int(row["row_id"]) for row in game_rows] != list(range(len(game_rows))):
        raise ValueError("Frozen v1 game row map is not contiguous")
    if [int(row["app_id"]) for row in game_rows] != [int(row["app_id"]) for row in descriptions]:
        raise ValueError("Frozen v1 descriptions do not follow the v1 game row mapping")
    return game_rows, descriptions


def _load_c_metadata(v1_rows: list[dict[str, Any]], split_path: Path = DEFAULT_SPLIT) -> dict[int, dict[str, Any]]:
    source = EDA / "legacy/data/processed/C_games_2025.parquet"
    app_path = split_path / "catalog/input_apps.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        rows = con.execute(f"""SELECT g.app_id,c.about_the_game,c.detailed_description,
            c.categories,c.developers,c.publishers
            FROM read_parquet({quote(app_path)}) g
            LEFT JOIN read_parquet({quote(source)}) c ON g.app_id=c.appid
            ORDER BY g.app_id""").fetchall()
    finally:
        con.close()
    if len(rows) != len(v1_rows):
        raise ValueError("C metadata join no longer matches the frozen v1 catalog")
    return {int(row[0]): {"about_the_game": row[1], "detailed_description": row[2],
                          "categories": row[3], "developers": row[4], "publishers": row[5]}
            for row in rows}


def audit_metadata_source_equivalence(selected: Path = DEFAULT_V2_METADATA,
                                     rebuilt: Path = V2_METADATA_REBUILT) -> dict[str, Any]:
    """Prove the former build path is byte-identical to the selected v2 source."""
    selected_manifest = read_json(selected / "manifest.json")
    rebuilt_manifest = read_json(rebuilt / "manifest.json")
    if selected_manifest.get("status") != "complete" or rebuilt_manifest.get("status") != "complete":
        raise ValueError("Both v2 metadata source artifacts must be complete for equivalence audit")
    compare_files = ("tfidf_csr.npz", "game_rows.parquet", "metadata_source_audit.parquet", "vocabulary_idf.parquet")
    file_results: dict[str, Any] = {}
    for name in compare_files:
        selected_hash = digest(selected / name)
        rebuilt_hash = digest(rebuilt / name)
        if selected_hash != rebuilt_hash:
            raise ValueError(f"v2 metadata source artifacts are not byte-identical: {name}")
        file_results[name] = {"sha256": selected_hash, "byte_identical": True}
    return {
        "status": "passed",
        "selected_v2_source": selected.relative_to(EDA).as_posix(),
        "selected_v2_manifest_sha256": digest(selected / "manifest.json"),
        "previous_builder_source": rebuilt.relative_to(EDA).as_posix(),
        "previous_builder_manifest_sha256": digest(rebuilt / "manifest.json"),
        "byte_identical_files": file_results,
        "reason": "The five preexisting v3 content artifacts remain immutable. Their metadata matrix was built from the prior rebuild path; byte-identical hashes prove the selected v2 matrix, row mapping, metadata features, and vocabulary are the same.",
    }


def _load_v2_metadata() -> tuple[dict[int, list[str]], dict[str, Any]]:
    folder = DEFAULT_V2_METADATA
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Frozen v2 metadata channel is not complete")
    for relative, info in manifest.get("files", {}).items():
        if digest(folder / relative) != info["sha256"]:
            raise ValueError(f"Frozen v2 metadata file changed: {relative}")
    rows = pq.read_table(folder / "metadata_source_audit.parquet",
                         columns=["app_id", "metadata_features"]).to_pylist()
    return ({int(row["app_id"]): list(row["metadata_features"] or []) for row in rows},
            {"directory": folder.relative_to(EDA).as_posix(),
             "manifest_sha256": digest(manifest_path),
             "metadata_source_audit_sha256": digest(folder / "metadata_source_audit.parquet")})


def _write_model(
    output: Path, model_id: str, config: dict[str, Any], build_fn: Callable[[Path, dict[str, Any]], None],
    *, protocol_path: Path = DEFAULT_PROTOCOL, split_path: Path = DEFAULT_SPLIT, v1_dir: Path = DEFAULT_V1,
) -> dict[str, Any]:
    target = output / model_id
    stage, manifest = _start_model(target, model_id, v1_dir, protocol_path, split_path)
    if stage == target:
        log(f"Verified existing v3 content model: {model_id}")
        return manifest
    manifest["parameters"] = config
    builder_snapshot = stage / "builder_snapshot.py"
    shutil.copyfile(Path(__file__).resolve(), builder_snapshot)
    manifest["lineage"]["builder_script"] = {
        "path": "builder_snapshot.py", "sha256": digest(builder_snapshot),
        "source_path": Path(__file__).resolve().relative_to(EDA).as_posix(),
    }
    try:
        build_fn(stage, manifest)
        result = _publish_model(stage, target, manifest)
        log(f"Published v3 content model: {model_id} {result['matrix'].get('shape')}")
        return result
    except Exception as error:
        _require_child(output, stage)
        log(f"Preserved incomplete content stage for resume: {stage}")
        raise RuntimeError(
            f"Building content model {model_id} failed: {type(error).__name__}: {error}; "
            f"incomplete stage preserved at {stage}"
        ) from error


def _transform_fit_sparse(
    folder: Path, docs: list[str], vectorizer: TfidfVectorizer, *, matrix_name: str = "tfidf_csr.npz",
    vocabulary_name: str = "vocabulary_idf.parquet",
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = vectorizer.fit_transform(docs).tocsr().astype(np.float32)
    matrix.sort_indices()
    matrix_info = _save_sparse_matrix(folder, matrix, matrix_name)
    _write_vocabulary(folder / vocabulary_name, vectorizer)
    return matrix_info, {"n_features": int(matrix.shape[1]), "nnz": int(matrix.nnz),
                         "nonzero_rows": int(np.count_nonzero(np.diff(matrix.indptr)))}


def _description_source_records(app_ids: list[int], sources: list[str], raw_texts: list[str], clean_texts: list[str]) -> list[dict[str, Any]]:
    return [{"row_id": i, "app_id": int(app_id), "source": source,
             "raw_char_count": len(raw_text or ""), "clean_char_count": len(clean_text or ""),
             "raw_sha256": sha_text(raw_text) if raw_text else None,
             "clean_sha256": sha_text(clean_text or "")}
            for i, (app_id, source, raw_text, clean_text) in enumerate(zip(app_ids, sources, raw_texts, clean_texts))]


def select_long_description(c_metadata: dict[str, Any], v1_raw_description: str | None) -> tuple[str | None, str]:
    for field in ("about_the_game", "detailed_description"):
        value = c_metadata.get(field)
        if has_raw_text(value):
            return value, f"C.{field}"
    if has_raw_text(v1_raw_description):
        return v1_raw_description, "frozen v1 A-first/C.short description"
    return None, "missing"


def select_frank_game_description(game: dict[str, Any] | None) -> tuple[str | None, str | None]:
    game = game or {}
    for field in ("about_the_game", "detailed_description", "short_description"):
        value = game.get(field)
        if has_raw_text(value):
            return value, field
    return None, None


def _build_text_model(
    stage: Path, manifest: dict[str, Any], app_ids: list[int], docs: list[str], vectorizer: TfidfVectorizer,
    *, source_records: list[dict[str, Any]] | None = None,
) -> None:
    matrix_info, summary = _transform_fit_sparse(stage, docs, vectorizer)
    manifest["matrix"] = matrix_info
    manifest["vectorizer"] = {"analyzer": vectorizer.analyzer, "ngram_range": list(vectorizer.ngram_range),
                              "min_df": vectorizer.min_df, "max_df": vectorizer.max_df,
                              "max_features": vectorizer.max_features, "sublinear_tf": vectorizer.sublinear_tf,
                              "use_idf": vectorizer.use_idf, "smooth_idf": vectorizer.smooth_idf,
                              "norm": vectorizer.norm, "dtype": "float32"}
    manifest["summary"] = {"games": len(app_ids), **summary}
    if source_records is not None:
        _write_source_audit(stage / "description_sources.parquet", source_records)


def _pinned_frank_games(path: Path = DEFAULT_FRANK_GAMES) -> tuple[Path, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(FRANK_GAMES_URL, timeout=60) as response, temporary.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > FRANK_GAMES_BYTES:
                    raise ValueError("Pinned frank games.parquet exceeded its declared size")
                out.write(chunk)
        temporary.replace(path)
    if path.stat().st_size != FRANK_GAMES_BYTES or digest(path) != FRANK_GAMES_SHA256:
        raise ValueError("Pinned frank games.parquet checksum or size differs from the fixed revision")
    return path, {"dataset": FRANK_DATASET, "commit": FRANK_COMMIT, "path": FRANK_GAMES_PATH,
                  "url": FRANK_GAMES_URL, "bytes": FRANK_GAMES_BYTES, "sha256": FRANK_GAMES_SHA256,
                  "local_path": path.relative_to(EDA).as_posix()}


def _load_frank_game_rows(path: Path, app_ids: set[int]) -> dict[int, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    needed = ["appid", "about_the_game", "detailed_description", "short_description"]
    out: dict[int, dict[str, Any]] = {}
    for batch in parquet.iter_batches(batch_size=50_000, columns=needed):
        app_column = batch.column("appid").to_pylist()
        for row, app in enumerate(app_column):
            if int(app) in app_ids:
                out[int(app)] = {name: batch.column(name)[row].as_py() for name in needed[1:]}
    return out


def _steam_appdetails(app_id: int, sources_dir: Path) -> tuple[str | None, dict[str, Any]]:
    """Fetch one small public Steam metadata response only as a frank fallback."""
    url = STEAM_APPDETAILS_URL.format(app_id=int(app_id))
    request = urllib.request.Request(url, headers={"User-Agent": "ML-game-recommendation content research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(STEAM_APPDETAILS_MAX_BYTES + 1)
    if len(body) > STEAM_APPDETAILS_MAX_BYTES:
        raise ValueError(f"Steam appdetails response exceeded the per-app size limit: {app_id}")
    parsed = json.loads(body)
    entry = parsed.get(str(app_id), {})
    data = entry.get("data") or {}
    path = sources_dir / f"steam_appdetails_{app_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    selected = next((data.get(field) for field in ("about_the_game", "detailed_description", "short_description")
                    if has_raw_text(data.get(field))), None)
    return selected, {"source": "Steam Store appdetails", "app_id": int(app_id), "url": url,
                      "retrieved_utc": utc_now(), "http_success": bool(entry.get("success")),
                      "selected_field": next((field for field in ("about_the_game", "detailed_description", "short_description")
                                              if has_raw_text(data.get(field))), None),
                      "file": path.relative_to(sources_dir.parent).as_posix(),
                      "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _fixed_fill_targets(game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                        v2_features: dict[int, list[str]]) -> list[int]:
    row_by_app = {int(row["app_id"]): row for row in game_rows}
    desc_by_app = {int(row["app_id"]): row for row in descriptions}
    expected = set(MISSING_DESCRIPTION_TARGET_APPS)
    if len(expected) != 10:
        raise ValueError("Fixed missing-description target set must contain exactly ten apps")
    for app_id in expected:
        row = row_by_app.get(app_id)
        desc = desc_by_app.get(app_id)
        if row is None or not row["linux"] or desc is None or desc["text_status"] == "supported":
            raise ValueError(f"A fixed missing-description target no longer matches the v1 Linux catalog: {app_id}")
        if v2_features.get(app_id):
            raise ValueError(f"A fixed fill target unexpectedly has v2 tag/genre metadata: {app_id}")
    return sorted(expected)


def build_missing_description_fill(output: Path, game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                                   v2_features: dict[int, list[str]], *, protocol_path: Path = DEFAULT_PROTOCOL,
                                   split_path: Path = DEFAULT_SPLIT, v1_dir: Path = DEFAULT_V1,
                                   model_id: str = "description_missing_fill") -> dict[str, Any]:
    if model_id not in {"description_missing_fill", "description_missing_fill_bigram"}:
        raise ValueError(f"Unsupported description fill model ID: {model_id}")
    ngram_range = (1, 2) if model_id.endswith("_bigram") else (1, 1)
    app_ids = [int(row["app_id"]) for row in game_rows]
    docs = [str(row["description_clean"] or "") for row in descriptions]
    raw_texts = [row["raw_description"] or "" for row in descriptions]
    desc_by_app = {int(row["app_id"]): row for row in descriptions}
    targets = _fixed_fill_targets(game_rows, descriptions, v2_features)
    frank_path, frank_provenance = _pinned_frank_games(output / "sources/frank_games.parquet")
    frank_rows = _load_frank_game_rows(frank_path, set(targets))
    sources_dir = output / "sources"
    missing_from_frank = sorted(set(targets) - frank_rows.keys())
    filled = []
    fallback_sources = []
    row_index = {app: i for i, app in enumerate(app_ids)}
    for app_id in targets:
        item = frank_rows.get(app_id, {})
        raw, field = select_frank_game_description(item)
        source = f"frank.games.parquet:{field}" if field else None
        if not has_raw_text(raw):
            raw, source_info = _steam_appdetails(app_id, sources_dir)
            fallback_sources.append(source_info)
            if has_raw_text(raw):
                source = f"Steam.appdetails:{source_info['selected_field']}"
        if not has_raw_text(raw):
            source = "still_missing_no_public_description"
            clean = ""
        else:
            clean = clean_description(raw)
            if not clean:
                source = "source_text_empty_after_cleaning"
        position = row_index[app_id]
        docs[position] = clean
        raw_texts[position] = raw or ""
        filled.append({"row_id": position, "app_id": app_id, "source": source,
                       "raw_char_count": len(raw or ""), "clean_char_count": len(clean),
                       "raw_sha256": sha_text(raw) if raw else None, "clean_sha256": sha_text(clean)})
    source_names = ["v1 frozen description" for _ in app_ids]
    for record in filled:
        source_names[record["row_id"]] = record["source"]
    records = _description_source_records(app_ids, source_names, raw_texts, docs)

    config = dict(MODEL_CONFIGS[model_id])
    config.update({"fixed_target_app_ids": targets, "target_count": len(targets),
                   "vectorizer_parameters": {"analyzer": "word", "tokenizer": "tfidf_content.tokenize_description",
                                               "ngram_range": list(ngram_range), "min_df": 1, "max_df": 1.0,
                                               "sublinear_tf": False, "norm": "l2", "dtype": "float32"},
                   "source_priority": ["pinned frank games.parquet about_the_game", "pinned frank games.parquet detailed_description",
                                       "pinned frank games.parquet short_description", "Steam appdetails about_the_game/detailed_description/short_description"],
                   "frank_source": frank_provenance,
                   "steam_fallback_apps": [int(x["app_id"]) for x in fallback_sources],
                   "frank_target_rows_found": len(set(targets) & frank_rows.keys()),
                   "target_apps_without_frank_row": missing_from_frank})

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        vectorizer = make_vectorizer() if ngram_range == (1, 1) else make_word_vectorizer(ngram_range)
        _build_text_model(stage, manifest, app_ids, docs, vectorizer, source_records=records)
        pq.write_table(pa.Table.from_pylist(filled), stage / "fill_audit.parquet", compression="zstd")
        manifest["parameters"] = config
        manifest["source_lineage"] = {"frank_games": frank_provenance,
                                      "steam_appdetails_fallbacks": fallback_sources,
                                      "filled_apps": filled}
        manifest["summary"]["filled_supported_apps"] = sum(bool(row["clean_char_count"]) for row in filled)
        manifest["summary"]["targets_remaining_unsupported"] = sum(not row["clean_char_count"] for row in filled)
    return _write_model(output, model_id, config, build, protocol_path=protocol_path,
                        split_path=split_path, v1_dir=v1_dir)


def build_long_descriptions(output: Path, game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                            c_metadata: dict[int, dict[str, Any]], *, protocol_path: Path = DEFAULT_PROTOCOL,
                            split_path: Path = DEFAULT_SPLIT, v1_dir: Path = DEFAULT_V1) -> dict[str, Any]:
    app_ids = [int(row["app_id"]) for row in game_rows]
    desc_by_app = {int(row["app_id"]): row for row in descriptions}
    docs: list[str] = []
    sources: list[str] = []
    raw_texts: list[str] = []
    for app_id in app_ids:
        c_row = c_metadata.get(app_id, {})
        raw, source = select_long_description(c_row, desc_by_app[app_id]["raw_description"])
        raw_texts.append(raw or "")
        docs.append(clean_description(raw))
        sources.append(source)
    records = _description_source_records(app_ids, sources, raw_texts, docs)
    config = dict(MODEL_CONFIGS["description_long_word_tfidf"])
    config["vectorizer_parameters"] = {"analyzer": "word", "tokenizer": "tfidf_content.tokenize_description",
                                       "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0,
                                       "sublinear_tf": False, "norm": "l2", "dtype": "float32"}

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        _build_text_model(stage, manifest, app_ids, docs, make_word_vectorizer((1, 2)), source_records=records)
        manifest["parameters"] = config
        manifest["source_lineage"]["C_games_2025"] = {
            "path": "legacy/data/processed/C_games_2025.parquet",
            "sha256": digest(EDA / "legacy/data/processed/C_games_2025.parquet"),
            "fields": ["about_the_game", "detailed_description"],
            "source_priority": ["C.about_the_game", "C.detailed_description", "frozen v1 description"],
        }
        manifest["summary"]["source_counts"] = {name: sources.count(name) for name in sorted(set(sources))}
    return _write_model(output, "description_long_word_tfidf", config, build, protocol_path=protocol_path,
                        split_path=split_path, v1_dir=v1_dir)


def _extra_metadata_features(c_row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    features: list[str] = []
    audit: dict[str, Any] = {}
    for field, prefix in (("categories", "category"), ("developers", "developer"), ("publishers", "publisher")):
        labels, status, parser = parse_metadata_labels(c_row.get(field), expected=field)
        normalized = sorted({normalize_concept(label) for label in labels if normalize_concept(label)})
        features.extend(f"{prefix}::{label}" for label in normalized)
        audit[field] = {"status": status, "parser": parser, "labels": normalized, "feature_count": len(normalized)}
    return sorted(set(features)), audit


def build_extended_metadata(output: Path, game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                            c_metadata: dict[int, dict[str, Any]], v2_features: dict[int, list[str]],
                            v2_provenance: dict[str, Any], *, protocol_path: Path = DEFAULT_PROTOCOL,
                            split_path: Path = DEFAULT_SPLIT, v1_dir: Path = DEFAULT_V1) -> dict[str, Any]:
    app_ids = [int(row["app_id"]) for row in game_rows]
    text_docs = [str(row["description_clean"] or "") for row in descriptions]
    metadata_docs: list[list[str]] = []
    audit_rows: list[dict[str, Any]] = []
    for row_id, app_id in enumerate(app_ids):
        base = list(v2_features.get(app_id, []))
        added, audit = _extra_metadata_features(c_metadata.get(app_id, {}))
        metadata_docs.append(sorted(set(base + added)))
        audit_rows.append({"row_id": row_id, "app_id": app_id,
                           "v2_tag_genre_feature_count": len(base),
                           "extended_category_feature_count": audit["categories"]["feature_count"],
                           "extended_developer_feature_count": audit["developers"]["feature_count"],
                           "extended_publisher_feature_count": audit["publishers"]["feature_count"],
                           "metadata_features": metadata_docs[-1], "parse_audit": json.dumps(audit, ensure_ascii=False)})
    config = dict(MODEL_CONFIGS["metadata_extended_tfidf"])
    config.update({"description_vectorizer": {"analyzer": "word", "tokenizer": "tfidf_content.tokenize_description",
                                               "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0,
                                               "sublinear_tf": False, "norm": "l2", "dtype": "float32"},
                   "metadata_vectorizer": {"binary": True, "use_idf": True, "norm": "l2", "dtype": "float32"},
                   "description_weight": 0.5, "metadata_weight": 0.5})

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        text_vectorizer = make_word_vectorizer((1, 2))
        text_info, text_summary = _transform_fit_sparse(stage, text_docs, text_vectorizer)
        metadata_vectorizer = make_metadata_vectorizer()
        metadata_matrix = metadata_vectorizer.fit_transform(metadata_docs).tocsr().astype(np.float32)
        metadata_matrix.sort_indices()
        metadata_info = _save_sparse_matrix(stage, metadata_matrix, "features_csr.npz")
        _write_vocabulary(stage / "metadata_vocabulary_idf.parquet", metadata_vectorizer)
        _write_vocabulary(stage / "vocabulary_idf.parquet", text_vectorizer)
        pq.write_table(pa.Table.from_pylist(audit_rows), stage / "metadata_extension_audit.parquet", compression="zstd")
        manifest["matrix"] = text_info
        manifest["components"] = [{"name": "description", **text_info, "weight": 0.5},
                                  {"name": "extended_metadata", "path": "features_csr.npz",
                                   "shape": list(metadata_matrix.shape), "nnz": int(metadata_matrix.nnz),
                                   "dtype": str(metadata_matrix.dtype), "format": "scipy CSR NPZ", "weight": 0.5}]
        manifest["score_spec"] = config["score_spec"]
        manifest["summary"] = {"games": len(app_ids), "description": text_summary,
                                "metadata": {"n_features": int(metadata_matrix.shape[1]),
                                             "nnz": int(metadata_matrix.nnz),
                                             "nonzero_rows": int(np.count_nonzero(np.diff(metadata_matrix.indptr)))}}
        manifest["parameters"] = config
        manifest["source_lineage"]["v2_tag_genre_metadata"] = v2_provenance
        manifest["source_lineage"]["C_extended_metadata"] = {
            "path": "legacy/data/processed/C_games_2025.parquet",
            "sha256": digest(EDA / "legacy/data/processed/C_games_2025.parquet"),
            "fields": ["categories", "developers", "publishers"],
            "feature_prefixes": {"categories": "category::", "developers": "developer::", "publishers": "publisher::"},
        }
    return _write_model(output, "metadata_extended_tfidf", config, build, protocol_path=protocol_path,
                        split_path=split_path, v1_dir=v1_dir)


def build_character_tfidf(output: Path, game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                          *, protocol_path: Path = DEFAULT_PROTOCOL, split_path: Path = DEFAULT_SPLIT,
                          v1_dir: Path = DEFAULT_V1) -> dict[str, Any]:
    app_ids = [int(row["app_id"]) for row in game_rows]
    docs = [str(row["description_clean"] or "") for row in descriptions]
    config = dict(MODEL_CONFIGS["description_char_wb_tfidf"])
    config["vectorizer_parameters"] = {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2,
                                       "max_df": 1.0, "max_features": 100_000, "sublinear_tf": True,
                                       "norm": "l2", "dtype": "float32"}

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        _build_text_model(stage, manifest, app_ids, docs, make_char_vectorizer())
        manifest["parameters"] = config
        manifest["source_lineage"]["description_cleaning"] = {
            "v1_manifest_sha256": digest(DEFAULT_V1 / "manifest.json"),
            "text_source": "frozen v1 descriptions.parquet description_clean column",
            "language_policy": "all scripts retained; character n-grams do not require language-specific tokenization",
        }
    return _write_model(output, "description_char_wb_tfidf", config, build, protocol_path=protocol_path,
                        split_path=split_path, v1_dir=v1_dir)


def build_sentence_embeddings(output: Path, game_rows: list[dict[str, Any]], descriptions: list[dict[str, Any]],
                               *, protocol_path: Path = DEFAULT_PROTOCOL, split_path: Path = DEFAULT_SPLIT,
                               v1_dir: Path = DEFAULT_V1) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    app_ids = [int(row["app_id"]) for row in game_rows]
    docs = [str(row["description_clean"] or "") for row in descriptions]
    nonempty = np.asarray([i for i, value in enumerate(docs) if value], dtype=np.int64)
    config = dict(MODEL_CONFIGS["description_sentence_embedding"])
    config.update({"device": "cpu", "torch_threads": 2, "max_seq_length": MINILM_MAX_LENGTH,
                   "batch_size": MINILM_BATCH_SIZE, "normalize_embeddings": True,
                   "empty_description_policy": "zero vector, excluded from encoder inference"})

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        started = time.perf_counter()
        model = SentenceTransformer(MINILM_MODEL, revision=MINILM_REVISION, device="cpu")
        model.max_seq_length = MINILM_MAX_LENGTH
        dimension = int(model.get_sentence_embedding_dimension())
        matrix = np.zeros((len(app_ids), dimension), dtype=np.float32)
        if len(nonempty):
            vectors = model.encode([docs[int(i)] for i in nonempty], batch_size=MINILM_BATCH_SIZE,
                                   show_progress_bar=True, convert_to_numpy=True,
                                   normalize_embeddings=True, device="cpu")
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(nonempty), dimension):
                raise ValueError("Sentence embedding model returned unexpected dimensions")
            matrix[nonempty] = vectors
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Sentence embeddings contain nonfinite values")
        norms = np.linalg.norm(matrix, axis=1)
        if len(nonempty) and not np.allclose(norms[nonempty], 1.0, atol=2e-4):
            raise ValueError("Nonempty sentence embeddings are not unit normalized")
        np.save(stage / "embeddings.npy", matrix, allow_pickle=False)
        manifest["matrix"] = {"path": "embeddings.npy", "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                               "format": "dense NumPy NPY sentence embeddings", "nonzero_rows": int(len(nonempty))}
        manifest["vectorizer"] = {"model_id": MINILM_MODEL, "revision": MINILM_REVISION,
                                  "inference": "SentenceTransformer.encode", "device": "CPU",
                                  "torch_threads": 2, "batch_size": MINILM_BATCH_SIZE,
                                  "max_seq_length": MINILM_MAX_LENGTH, "normalize_embeddings": True,
                                  "tokenization_truncation": "inputs are truncated at model max_seq_length=128",
                                  "trained_or_fine_tuned_on_current_review_labels": False}
        manifest["summary"] = {"games": len(app_ids), "embedded_games": int(len(nonempty)),
                                "zero_vectors": int(len(app_ids) - len(nonempty)), "dimension": dimension,
                                "encoder_elapsed_seconds": round(time.perf_counter() - started, 3)}
        manifest["source_lineage"]["pretrained_model"] = {"repo_id": MINILM_MODEL,
                                                            "revision": MINILM_REVISION,
                                                            "model_card_url": f"https://huggingface.co/{MINILM_MODEL}/tree/{MINILM_REVISION}",
                                                            "model_file_size_checked_before_inference": True,
                                                            "model_safetensors_bytes": 470_641_600}
    return _write_model(output, "description_sentence_embedding", config, build,
                        protocol_path=protocol_path, split_path=split_path, v1_dir=v1_dir)


def _sorted_train_review_keys(split_path: Path, work_dir: Path) -> tuple[np.memmap, Path, int]:
    lineage_path = split_path / "train/review_lineage.parquet"
    sorted_path = work_dir / "accepted_train_review_keys_sorted.parquet"
    keys_path = work_dir / "accepted_train_review_keys.npy"
    split_manifest = read_json(split_path / "manifest.json")
    lineage_info = split_manifest.get("files", {}).get("train/review_lineage.parquet", {})
    if not lineage_path.is_file() or digest(lineage_path) != lineage_info.get("sha256"):
        raise ValueError("Accepted train review lineage checksum differs from the frozen split manifest")
    lineage_rows = int(lineage_info.get("rows", pq.ParquetFile(lineage_path).metadata.num_rows))
    if keys_path.is_file():
        cached = np.load(keys_path, mmap_mode="r", allow_pickle=False)
        if cached.dtype != np.uint64 or cached.shape != (lineage_rows,):
            del cached
            raise ValueError("Cached accepted-train review ID map has an unexpected shape or dtype")
        unique_count = 0
        previous: int | None = None
        padding_started = False
        for start in range(0, lineage_rows, 1_000_000):
            values = cached[start:min(start + 1_000_000, lineage_rows)]
            nonzero = values[values != 0]
            if len(nonzero):
                if padding_started:
                    del cached
                    raise ValueError("Cached accepted-train review ID map has a nonzero value after padding")
                if previous is not None and int(nonzero[0]) <= previous:
                    del cached
                    raise ValueError("Cached accepted-train review ID map is not strictly sorted")
                if len(nonzero) > 1 and np.any(nonzero[1:] <= nonzero[:-1]):
                    del cached
                    raise ValueError("Cached accepted-train review ID map is not strictly sorted")
                previous = int(nonzero[-1])
                unique_count += len(nonzero)
            if len(nonzero) < len(values):
                padding_started = True
        if unique_count < 1:
            del cached
            raise ValueError("Cached accepted-train review ID map is empty")
        log(f"Reusing cached accepted training review IDs: {unique_count:,} unique IDs")
        return cached[:unique_count], keys_path, unique_count
    sorted_path.unlink(missing_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        # The prior 1GB attempt built a global DISTINCT hash table before its
        # sort and failed. A sorted external pass plus streaming dedup avoids
        # that hash table and stays within the content worker's 2GB budget.
        con.execute("SET memory_limit='1GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={quote(work_dir / 'duckdb_tmp')}")
        con.execute(f"SET max_temp_directory_size='100GB'")
        log(f"External sorting {lineage_rows:,} accepted training review IDs with 1GB DuckDB cap")
        con.execute(f"""COPY (SELECT review_key FROM read_parquet({quote(lineage_path)})
            ORDER BY review_key) TO {quote(sorted_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)""")
    finally:
        con.close()
    if pq.ParquetFile(sorted_path).metadata.num_rows != lineage_rows:
        raise ValueError("Sorted accepted train review ID table row count changed")
    keys = np.lib.format.open_memmap(keys_path, mode="w+", dtype=np.uint64, shape=(lineage_rows,))
    offset = 0
    previous = None
    for batch in pq.ParquetFile(sorted_path).iter_batches(batch_size=1_000_000, columns=["review_key"]):
        values = batch.column("review_key").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        if not len(values):
            continue
        keep = np.ones(len(values), dtype=bool)
        if len(values) > 1:
            keep[1:] = values[1:] != values[:-1]
        if previous is not None and int(values[0]) == previous:
            keep[0] = False
        unique_values = values[keep]
        next_offset = offset + len(unique_values)
        keys[offset:next_offset] = unique_values
        offset = next_offset
        previous = int(values[-1])
    keys.flush()
    if offset < 1:
        raise ValueError("No accepted training review IDs were found")
    unique_keys = keys[:offset]
    sorted_path.unlink()
    if offset > 1 and not int(unique_keys[0]) < int(unique_keys[-1]):
        raise ValueError("Accepted train review IDs did not sort into a strictly increasing range")
    return unique_keys, keys_path, offset


def _accepted_key_mask(review_ids: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_keys, review_ids)
    found = positions < len(sorted_keys)
    if np.any(found):
        found_indices = np.flatnonzero(found)
        found[found_indices] = sorted_keys[positions[found_indices]] == review_ids[found_indices]
    return found


def _verify_sampled_train_lineage(sample_path: Path, lineage_path: Path, work_dir: Path,
                                  expected_sample_count: int) -> int:
    """Verify every sampled (review, user, app) tuple with a bounded join."""
    (work_dir / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={quote(work_dir / 'duckdb_tmp')}")
        con.execute("SET max_temp_directory_size='100GB'")
        # Put the small sampled relation on the hash-build side; the accepted
        # lineage scan streams under DuckDB's 1GB limit.
        matched = int(con.execute(f"""SELECT count(DISTINCT s.review_key)
            FROM read_parquet({quote(lineage_path)}) l
            INNER JOIN read_parquet({quote(sample_path)}) s
            ON l.review_key=s.review_key AND l.user_key=s.user_key AND l.app_id=s.app_id""").fetchone()[0])
    finally:
        con.close()
    if matched != expected_sample_count:
        raise ValueError(
            "Sampled review bodies do not match accepted train review_key/user_key/app_id lineage "
            f"({matched:,} of {expected_sample_count:,} sampled IDs matched)"
        )
    return matched


def _expected_train_review_row_count(ingestion: dict[str, Any]) -> int:
    """Read the raw row count from either supported Kieran manifest schema."""
    for field in ("raw_data_rows_parsed", "source_rows_expected_from_dataset_card", "canonical_rows_written"):
        value = ingestion.get(field)
        if value is not None:
            count = int(value)
            if count >= 0:
                return count
    raise ValueError("Kieran ingestion manifest is missing its raw source row count")


def _completed_review_scan_info(sample_path: Path, lineage_sample_path: Path, marker_path: Path,
                                expected_raw_rows: int) -> dict[str, Any] | None:
    """Recognize a complete prior scan while rejecting truncated Parquet files."""
    if not sample_path.is_file():
        return None
    try:
        sample_file = pq.ParquetFile(sample_path)
        sampled_count = int(sample_file.metadata.num_rows)
        if sampled_count < 1 or "review_text" not in sample_file.schema.names:
            return None
        if marker_path.is_file():
            marker = read_json(marker_path)
            if (int(marker.get("raw_csv_rows_scanned", -1)) != expected_raw_rows
                    or int(marker.get("sampled_unique_review_keys", -1)) != sampled_count
                    or marker.get("sample_parquet_sha256") != digest(sample_path)):
                return None
            return {"raw_csv_rows_scanned": expected_raw_rows,
                    "accepted_train_catalog_review_rows_seen": marker.get("accepted_train_catalog_review_rows_seen"),
                    "sampled_unique_review_keys": sampled_count}
        # Older interrupted runs had no marker, but wrote this lineage copy
        # only after verifying the complete source row count.
        if lineage_sample_path.is_file():
            lineage_count = int(pq.ParquetFile(lineage_sample_path).metadata.num_rows)
            if lineage_count == sampled_count:
                return {"raw_csv_rows_scanned": expected_raw_rows,
                        "accepted_train_catalog_review_rows_seen": None,
                        "sampled_unique_review_keys": sampled_count}
    except (OSError, ValueError, KeyError, pa.ArrowException):
        return None
    return None


def _scan_kieran_review_csv(archive_path: Path, app_ids: list[int], train_keys: np.ndarray,
                            sample_path: Path, ingestion: dict[str, Any], expected_bytes: int) -> dict[str, Any]:
    """Extract only the capped review bodies whose IDs belong to train lineage."""
    sample_schema = pa.schema([
        ("app_id", pa.int64()), ("review_key", pa.uint64()), ("user_key", pa.uint64()),
        ("source_row_number", pa.uint64()), ("sample_rank", pa.uint16()), ("review_text", pa.large_string()),
    ])
    sample_writer = pq.ParquetWriter(sample_path, sample_schema, compression="zstd", use_dictionary=False)
    buffer: list[tuple[int, int, int, int, int, str]] = []
    sample_counts = np.zeros(len(app_ids), dtype=np.uint16)
    sampled_keys: set[int] = set()
    total_rows = 0
    accepted_train_catalog_rows = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.getinfo("all_reviews/all_reviews.csv")
            if member.file_size != expected_bytes:
                raise ValueError(f"Unexpected Kieran CSV size inside archive: {member.file_size}")
            with archive.open(member) as stream:
                reader = pacsv.open_csv(
                    stream,
                    read_options=pacsv.ReadOptions(block_size=KIERAN_CSV_BLOCK_BYTES, use_threads=False),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=["recommendationid", "appid", "author_steamid", "review"],
                        column_types={"recommendationid": pa.uint64(), "appid": pa.int64(),
                                      "author_steamid": pa.uint64(), "review": pa.large_string()},
                    ),
                )
                while True:
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    batch_size = batch.num_rows
                    if batch_size == 0:
                        continue
                    row_numbers = np.arange(total_rows, total_rows + batch_size, dtype=np.uint64)
                    total_rows += batch_size
                    review_ids = batch.column("recommendationid").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                    apps = batch.column("appid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                    users = batch.column("author_steamid").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                    accepted = _accepted_key_mask(review_ids, train_keys)
                    positions = np.searchsorted(np.asarray(app_ids, dtype=np.int64), apps)
                    in_catalog = positions < len(app_ids)
                    valid_catalog = np.flatnonzero(in_catalog)
                    in_catalog[valid_catalog] = np.asarray(app_ids, dtype=np.int64)[positions[valid_catalog]] == apps[valid_catalog]
                    chosen = np.flatnonzero(accepted & in_catalog)
                    accepted_train_catalog_rows += int(len(chosen))
                    for j in chosen:
                        review_key = int(review_ids[j])
                        app_idx = int(positions[j])
                        user_id = int(users[j])
                        if not STEAM_BASE <= user_id <= STEAM_BASE + (2**32 - 1):
                            raise ValueError("Sampled Kieran author Steam ID is outside the validated range")
                        if review_key in sampled_keys or sample_counts[app_idx] >= REVIEW_SAMPLE_CAP:
                            continue
                        sampled_keys.add(review_key)
                        sample_counts[app_idx] += 1
                        sample_rank = int(sample_counts[app_idx])
                        buffer.append((int(apps[j]), review_key, user_id, int(row_numbers[j]), sample_rank,
                                       batch.column("review")[int(j)].as_py() or ""))
                        if len(buffer) >= 5000:
                            sample_writer.write_table(pa.Table.from_pylist([
                                {"app_id": r[0], "review_key": r[1], "user_key": r[2],
                                 "source_row_number": r[3], "sample_rank": r[4], "review_text": r[5]}
                                for r in buffer], schema=sample_schema), row_group_size=5000)
                            buffer.clear()
                    if total_rows % 1_000_000 < batch_size:
                        log(f"Scanning Kieran train review bodies: {total_rows:,} rows; sampled {len(sampled_keys):,}")
        if buffer:
            sample_writer.write_table(pa.Table.from_pylist([
                {"app_id": r[0], "review_key": r[1], "user_key": r[2],
                 "source_row_number": r[3], "sample_rank": r[4], "review_text": r[5]}
                for r in buffer], schema=sample_schema), row_group_size=5000)
            buffer.clear()
    finally:
        sample_writer.close()
    expected_raw_rows = _expected_train_review_row_count(ingestion)
    if total_rows != expected_raw_rows:
        raise ValueError(f"Kieran raw CSV row count changed: {total_rows} != {expected_raw_rows}")
    sampled_count = int(pq.ParquetFile(sample_path).metadata.num_rows)
    if sampled_count != len(sampled_keys) or int(sample_counts.sum()) != sampled_count:
        raise ValueError("Sampled review body counts and persisted review sample rows disagree")
    return {"raw_csv_rows_scanned": total_rows,
            "accepted_train_catalog_review_rows_seen": accepted_train_catalog_rows,
            "sampled_unique_review_keys": sampled_count}


def _write_review_docs(stage: Path, raw_documents_path: Path, app_ids: list[int]) -> tuple[list[str], dict[str, Any]]:
    """Clean aggregated documents in batches and retain only hashes and counts."""
    app_position = {int(app): i for i, app in enumerate(app_ids)}
    docs = [""] * len(app_ids)
    counts = np.zeros(len(app_ids), dtype=np.uint16)
    parquet = pq.ParquetFile(raw_documents_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["app_id", "sample_review_count", "raw_document"]):
        apps = batch.column("app_id").to_pylist()
        sample_counts = batch.column("sample_review_count").to_pylist()
        raw_documents = batch.column("raw_document").to_pylist()
        for app_id, count, raw in zip(apps, sample_counts, raw_documents):
            row_id = app_position.get(int(app_id))
            if row_id is None:
                raise ValueError(f"Aggregated review document is outside the fixed app catalog: {app_id}")
            counts[row_id] = int(count)
            docs[row_id] = clean_description(raw or "")
    audit = pa.table({
        "row_id": pa.array(np.arange(len(app_ids), dtype=np.int64)),
        "app_id": pa.array(np.asarray(app_ids, dtype=np.int64)),
        "sample_review_count": pa.array(counts, type=pa.uint16()),
        "clean_char_count": pa.array([len(doc) for doc in docs], type=pa.int64()),
        "clean_document_sha256": pa.array([sha_text(doc) for doc in docs], type=pa.string()),
    })
    pq.write_table(audit, stage / "train_review_document_audit.parquet", compression="zstd")
    hist: dict[str, int] = {}
    for value in np.unique(counts):
        hist[str(int(value))] = int(np.count_nonzero(counts == value))
    return docs, {"games_with_sampled_reviews": int(np.count_nonzero(counts)),
                  "sampled_review_count": int(counts.sum()), "max_reviews_per_app": int(counts.max(initial=0)),
                  "review_count_histogram": hist}


def build_train_review_body(output: Path, game_rows: list[dict[str, Any]], *, split_path: Path = DEFAULT_SPLIT,
                            protocol_path: Path = DEFAULT_PROTOCOL, v1_dir: Path = DEFAULT_V1,
                            archive_path: Path = DEFAULT_KIERAN_ARCHIVE,
                            ingestion_manifest_path: Path = DEFAULT_KIERAN_INGESTION) -> dict[str, Any]:
    app_ids = [int(row["app_id"]) for row in game_rows]
    app_position = {app: i for i, app in enumerate(app_ids)}
    config = dict(MODEL_CONFIGS["train_review_body_tfidf"])
    config.update({"sample_cap_per_item": REVIEW_SAMPLE_CAP,
                   "sample_selection": REVIEW_SAMPLE_POLICY,
                   "word_vectorizer_parameters": {"analyzer": "word", "tokenizer": "tfidf_content.tokenize_description",
                                                   "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0,
                                                   "sublinear_tf": False, "norm": "l2", "dtype": "float32"},
                   "train_account_or_review_filter": "review_key membership in split_v1/train/review_lineage.parquet; sampled rows additionally checked against review_key, user_key, app_id lineage"})

    def build(stage: Path, manifest: dict[str, Any]) -> None:
        ingestion = read_json(ingestion_manifest_path)
        archive_info = read_json(DEFAULT_SOURCE_MANIFEST)["datasets"]["kieran"]
        if not archive_path.is_file() or digest(archive_path) != archive_info["archive_sha256"]:
            raise ValueError("Kieran source archive is missing or differs from the pinned source SHA256")
        if ingestion.get("archive_sha256") != archive_info["archive_sha256"]:
            raise ValueError("Kieran ingestion manifest does not match the pinned source archive")
        archive_sha = digest(archive_path)
        work_dir = stage / ".review_work"
        work_dir.mkdir(exist_ok=True)
        train_keys, keys_path, train_unique_ids = _sorted_train_review_keys(split_path, work_dir)
        sample_temp = work_dir / "train_review_sample.parquet"
        expected_raw_rows = _expected_train_review_row_count(ingestion)
        sample_lineage_path = stage / "review_sample_lineage.parquet"
        review_documents_path = stage / "review_documents_raw.parquet"
        scan_marker_path = work_dir / "review_scan_complete.json"
        lineage_path = split_path / "train/review_lineage.parquet"
        expected_bytes = 42_487_005_552
        scan_info = _completed_review_scan_info(sample_temp, sample_lineage_path, scan_marker_path,
                                                expected_raw_rows)
        if scan_info is None:
            # Discard a truncated sample from an interrupted CSV reader; the
            # accepted train key map remains cached for the replacement scan.
            sample_temp.unlink(missing_ok=True)
            sample_lineage_path.unlink(missing_ok=True)
            review_documents_path.unlink(missing_ok=True)
            scan_info = _scan_kieran_review_csv(archive_path, app_ids, train_keys, sample_temp,
                                                ingestion, expected_bytes)
            write_json(scan_marker_path, {**scan_info, "sample_parquet_sha256": digest(sample_temp)})
        else:
            log(f"Reusing complete sampled train review scan: {scan_info['sampled_unique_review_keys']:,} rows")
        total_rows = int(scan_info["raw_csv_rows_scanned"])
        accepted_train_catalog_rows = scan_info.get("accepted_train_catalog_review_rows_seen")
        sampled_count = int(scan_info["sampled_unique_review_keys"])
        review_documents_path.unlink(missing_ok=True)
        con = duckdb.connect()
        try:
            con.execute("SET threads=2")
            con.execute("SET memory_limit='1GB'")
            con.execute(f"SET temp_directory={quote(work_dir / 'duckdb_tmp')}")
            con.execute(f"SET max_temp_directory_size='100GB'")
            persisted_sample_count = int(con.execute(f"SELECT count(*) FROM read_parquet({quote(sample_temp)})").fetchone()[0])
            if persisted_sample_count != sampled_count:
                raise ValueError("Persisted review sample count differs from the completed scan marker")
            sample_lineage_path.unlink(missing_ok=True)
            con.execute(f"""COPY (SELECT app_id,review_key,source_row_number,sample_rank
                FROM read_parquet({quote(sample_temp)}) ORDER BY app_id,source_row_number)
                TO {quote(sample_lineage_path)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
            con.execute("SET memory_limit='3GB'")
            con.execute(f"""COPY (SELECT app_id,count(*) sample_review_count,
                string_agg(review_text,' ' ORDER BY source_row_number) raw_document
                FROM read_parquet({quote(sample_temp)}) GROUP BY app_id ORDER BY app_id)
                TO {quote(review_documents_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)""")
        finally:
            con.close()
        # Close and remove the memory-mapped accepted-key cache before the
        # lineage verification join, keeping both checks bounded independently.
        train_keys._mmap.close()
        del train_keys
        gc.collect()
        keys_path.unlink()
        if keys_path.exists():
            raise PermissionError("Accepted-train review ID map remained locked after closing its memory map")
        sampled_lineage_checks = _verify_sampled_train_lineage(
            sample_temp, lineage_path, work_dir, sampled_count
        )
        docs, sample_summary = _write_review_docs(stage, review_documents_path, app_ids)
        review_documents_path.unlink()
        sample_temp.unlink()
        gc.collect()
        _require_child(stage, work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        if work_dir.exists():
            raise PermissionError("Review extraction work directory remained locked after closing the memory maps")
        if keys_path.exists():
            raise ValueError("Temporary accepted-train review ID map was not removed")

        matrix_info, summary = _transform_fit_sparse(stage, docs, make_word_vectorizer((1, 2)))
        manifest["matrix"] = matrix_info
        manifest["vectorizer"] = {"analyzer": "word", "tokenizer": "tfidf_content.tokenize_description",
                                  "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0,
                                  "sublinear_tf": False, "use_idf": True, "norm": "l2", "dtype": "float32"}
        manifest["parameters"] = config
        manifest["summary"] = {"games": len(app_ids), **summary, **sample_summary,
                                "accepted_train_review_ids": int(train_unique_ids),
                                "raw_csv_rows_scanned": total_rows,
                                "accepted_train_catalog_review_rows_seen": accepted_train_catalog_rows,
                                "sample_lineage_mismatches": 0,
                                "sample_lineage_rows_verified": sampled_lineage_checks}
        manifest["source_lineage"]["Kieran_original"] = {
            "dataset": "kieranpoc/steam-reviews", "version": 2, "archive_sha256": archive_sha,
            "archive_path": archive_path.relative_to(EDA).as_posix(),
            "zip_member": "all_reviews/all_reviews.csv", "csv_uncompressed_bytes": expected_bytes,
            "csv_parser_block_size_bytes": KIERAN_CSV_BLOCK_BYTES,
            "csv_rows": total_rows, "accepted_train_lineage_path": lineage_path.relative_to(EDA).as_posix(),
            "accepted_train_lineage_sha256": digest(lineage_path),
            "sampling": REVIEW_SAMPLE_POLICY, "sampled_unique_review_keys": sampled_count,
            "sampled_lineage_file": "review_sample_lineage.parquet",
            "sampled_lineage_mismatches": 0,
            "sampled_lineage_rows_verified": sampled_lineage_checks,
            "review_body_column": "review",
            "non_train_review_text_retained_or_fitted": False,
            "review_text_selection": "Only selected rows with accepted train review IDs and exact user/app lineage are copied to the sample file; other CSV batch text is discarded and never fitted.",
        }
    return _write_model(output, "train_review_body_tfidf", config, build, protocol_path=protocol_path,
                        split_path=split_path, v1_dir=v1_dir)


def build_all(output: Path = DEFAULT_OUTPUT, *, models: list[str] | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = _validate_sources()
    game_rows, descriptions = _load_v1_rows()
    app_ids = [int(row["app_id"]) for row in game_rows]
    c_metadata = _load_c_metadata(game_rows)
    v2_features, v2_provenance = _load_v2_metadata()
    metadata_equivalence = audit_metadata_source_equivalence()
    target_models = list(MODEL_CONFIGS) if models is None else list(models)
    unknown = set(target_models) - MODEL_CONFIGS.keys()
    if unknown:
        raise ValueError(f"Unknown v3 content models: {sorted(unknown)}")
    registry_path = output / "registry.json"
    previous_registry = read_json(registry_path) if registry_path.is_file() else {}
    registry: dict[str, Any] = {
        "version": VERSION, "status": "building", "created_utc": utc_now(),
        "protocol_sha256": sources["protocol_sha256"],
        "model_registry": [], "ranking_metrics_computed": False,
        "validation_or_test_labels_loaded_for_fitting": False,
        "builder_script": {"path": Path(__file__).resolve().relative_to(EDA).as_posix(),
                           "sha256": digest(Path(__file__).resolve())},
        "metadata_v2_source_equivalence_audit": metadata_equivalence,
        "standard_terms": {"family": "content", "vector_rows": "frozen v1 app_id/Linux mapping"},
    }
    if previous_registry.get("protocol_sha256") not in (None, sources["protocol_sha256"]):
        raise ValueError("Existing v3 content registry uses a different frozen protocol")
    for entry in previous_registry.get("model_registry", []):
        model_id = entry.get("model_id")
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Existing v3 content registry has an unknown model: {model_id}")
        model_folder = output / model_id
        manifest_path = model_folder / "manifest.json"
        if not manifest_path.is_file() or digest(manifest_path) != entry.get("manifest_sha256"):
            raise ValueError(f"Existing v3 content registry manifest changed: {model_id}")
        verify_model_artifact(model_folder, expected_rows=37_610)
        registry["model_registry"].append(entry)
    registry["model_registry"].sort(key=lambda x: x["model_id"])
    write_json(registry_path, registry)
    builders: dict[str, Callable[[], dict[str, Any]]] = {
        "description_missing_fill": lambda: build_missing_description_fill(output, game_rows, descriptions, v2_features),
        "description_missing_fill_bigram": lambda: build_missing_description_fill(
            output, game_rows, descriptions, v2_features, model_id="description_missing_fill_bigram"),
        "description_long_word_tfidf": lambda: build_long_descriptions(output, game_rows, descriptions, c_metadata),
        "metadata_extended_tfidf": lambda: build_extended_metadata(output, game_rows, descriptions, c_metadata,
                                                                      v2_features, v2_provenance),
        "description_char_wb_tfidf": lambda: build_character_tfidf(output, game_rows, descriptions),
        "description_sentence_embedding": lambda: build_sentence_embeddings(output, game_rows, descriptions),
        "train_review_body_tfidf": lambda: build_train_review_body(output, game_rows),
    }
    registry["source_manifest_sha256"] = sources["source_manifest_sha256"]
    registry["protocol"] = read_json(DEFAULT_PROTOCOL)
    registry["models_requested"] = target_models
    write_json(registry_path, registry)
    try:
        with threadpool_limits(limits=2):
            for model_id in target_models:
                result = builders[model_id]()
                model_folder = output / model_id
                entry = {"model_id": model_id, "family": "content",
                         "directory": model_folder.relative_to(EDA).as_posix(),
                         "manifest_sha256": digest(model_folder / "manifest.json"),
                         "parameters": result.get("parameters"),
                         "matrix": result.get("matrix"),
                         "components": result.get("components", [])}
                if not any(x["model_id"] == model_id for x in registry["model_registry"]):
                    registry["model_registry"].append(entry)
                else:
                    registry["model_registry"] = [entry if x["model_id"] == model_id else x for x in registry["model_registry"]]
                registry["model_registry"].sort(key=lambda x: x["model_id"])
                registry["updated_utc"] = utc_now()
                registry["files"] = file_inventory(output)
                write_json(registry_path, registry)
        registry["status"] = "complete"
        registry["completed_utc"] = utc_now()
        registry["files"] = file_inventory(output)
        write_json(registry_path, registry)
    except Exception as error:
        registry["status"] = "failed"
        registry["error"] = f"{type(error).__name__}: {error}"
        registry["updated_utc"] = utc_now()
        registry["files"] = file_inventory(output)
        write_json(registry_path, registry)
        raise
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_CONFIGS),
                        help="Build one model; repeat to build multiple. The default builds all registered content models.")
    args = parser.parse_args()
    result = build_all(args.output, models=args.model)
    print(json.dumps({"status": result["status"], "models": [m["model_id"] for m in result["model_registry"]],
                      "ranking_metrics_computed": result["ranking_metrics_computed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
