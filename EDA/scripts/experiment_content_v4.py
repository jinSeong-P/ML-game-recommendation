"""Build fixed Qwen3 game-content embeddings and fit KNN-aligned projectors.

This script uses only the frozen v1 description rows, the frozen v3 extended
metadata audit, the fixed v4 positive-support vector, and the fresh v4 KNN
teacher. It never reads validation or confirmation outcomes and computes no
recommendation metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse


EDA = Path(__file__).resolve().parents[1]
ROOT = EDA / "data/processed/modeling/experiments_v4"
CONTENT_ROOT = ROOT / "content"
DESCRIPTION_SOURCE = EDA / "data/processed/modeling/tfidf_content_v1/descriptions.parquet"
METADATA_SOURCE = EDA / "data/processed/modeling/improvement_v3/content/metadata_extended_tfidf/metadata_extension_audit.parquet"
SUPPORT_PATH = ROOT / "models/positive_support.npy"
TEACHER_PATH = ROOT / "models/activity_inverse_degree_cosine_k100/similarities_csr.npz"
CATALOG_PATH = ROOT / "split/catalog/input_apps.parquet"

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_LENGTHS = (512, 1024)
CHANNELS = ("description", "metadata")
EMBED_BATCH_SIZE = 8
ALIGNED_DIMS = (64, 128)
TEACHER_TOP_K = (20, 50)
NEGATIVE_RATIO = 2
WEAK_NEGATIVE_WEIGHT = 0.25
WEAK_NEGATIVE_MARGIN = 0.05
ALIGNMENT_EPOCHS = 8
ALIGNMENT_BATCH_SIZE = 4096
ALIGNMENT_SEED = 20261108


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Windows scanners can briefly hold an existing checkpoint while it is
    # being replaced. Retry the atomic rename without weakening file contents.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_content_texts() -> tuple[np.ndarray, dict[str, list[str]], dict[str, str]]:
    descriptions = pq.read_table(DESCRIPTION_SOURCE, columns=["row_id", "app_id", "description_clean"])
    metadata = pq.read_table(METADATA_SOURCE, columns=["row_id", "app_id", "metadata_features"])
    if descriptions.num_rows != 37_610 or metadata.num_rows != 37_610:
        raise ValueError("Frozen v1 description and metadata sources must each have 37,610 rows")
    d = descriptions.to_pydict()
    m = metadata.to_pydict()
    row_ids = np.arange(37_610, dtype=np.int64)
    if not np.array_equal(np.asarray(d["row_id"], dtype=np.int64), row_ids):
        raise ValueError("Description row IDs no longer match the frozen catalog order")
    if not np.array_equal(np.asarray(m["row_id"], dtype=np.int64), row_ids):
        raise ValueError("Metadata row IDs no longer match the frozen catalog order")
    app_ids = np.asarray(d["app_id"], dtype=np.int64)
    if not np.array_equal(app_ids, np.asarray(m["app_id"], dtype=np.int64)) or np.any(np.diff(app_ids) <= 0):
        raise ValueError("Description and metadata app IDs must share the same ascending row order")
    description_text = [str(value or "") for value in d["description_clean"]]
    metadata_text = [" ; ".join(sorted(set(value or []))) for value in m["metadata_features"]]
    source_hashes = {"descriptions_parquet_sha256": digest(DESCRIPTION_SOURCE),
                     "metadata_extension_audit_sha256": digest(METADATA_SOURCE)}
    return app_ids, {"description": description_text, "metadata": metadata_text}, source_hashes


def _embedding_name(channel: str, max_length: int) -> str:
    return f"qwen3_embedding_0_6b_{channel}_{max_length}"


def _manifest_for_embedding(channel: str, max_length: int, source_hashes: dict[str, str], dimension: int) -> dict[str, Any]:
    return {
        "status": "complete",
        "experiment": "V4-6 fixed Qwen3 Embedding content channel",
        "name": _embedding_name(channel, max_length),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "pooling": "last non-padding token (left padding; final sequence position)",
                  "normalization": "L2", "input_prompt": None},
        "channel": channel,
        "source_policy": {
            "description": "Frozen v1 descriptions.parquet description_clean (A.description first; C.short_description fallback already applied).",
            "metadata": "Frozen v3 metadata_extension_audit.parquet metadata_features, unique sorted feature labels joined with ' ; '.",
            "empty_text": "All-zero vector; empty rows bypass model output semantically.",
            "max_token_length_including_special_tokens": max_length,
            "truncation": "right truncation by the tokenizer",
        }[channel],
        "source_files": source_hashes,
        "catalog": {"rows": 37_610, "row_order": "frozen v1 ascending app_id", "app_ids_sha256": None},
        "embedding": {"dimension": dimension, "dtype": "float32", "shape": [37_610, dimension],
                      "batch_size": EMBED_BATCH_SIZE},
        "fit_data": "No review labels or evaluation cohorts were loaded.",
        "ranking_metrics_computed": False,
    }


def _tokenization_audit(tokenizer, text: list[str]) -> dict[str, int]:
    lengths: list[int] = []
    for begin in range(0, len(text), 64):
        encoded = tokenizer(text[begin:begin + 64], padding=False, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in encoded.input_ids)
    return {"rows": len(lengths), "empty_rows": int(sum(not value.strip() for value in text)),
            "max_tokens_including_special_tokens": max(lengths, default=0),
            "rows_over_512_tokens": int(sum(value > 512 for value in lengths)),
            "rows_over_1024_tokens": int(sum(value > 1024 for value in lengths))}


def _load_qwen(cache_dir: Path):
    # Set cache before importing huggingface_hub/transformers so all model
    # files remain inside the ignored EDA processed-data tree.
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "hub")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 embedding generation requires the prepared CUDA environment")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
                                               use_fast=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir), trust_remote_code=False,
        dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model.eval().to("cuda")
    model.config.use_cache = False
    return torch, F, tokenizer, model


def _verify_existing_embedding(folder: Path, expected: dict[str, Any]) -> bool:
    manifest_path = folder / "manifest.json"
    vectors_path = folder / "embeddings.npy"
    if not manifest_path.is_file() or not vectors_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "complete" or manifest.get("config_sha256") != expected["config_sha256"]:
        raise ValueError(f"Existing Qwen embedding artifact has a changed configuration: {folder.name}")
    if digest(vectors_path) != manifest["files"]["embeddings.npy"]["sha256"]:
        raise ValueError(f"Existing Qwen embedding artifact failed its file hash: {folder.name}")
    values = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if values.shape != tuple(expected["embedding"]["shape"]) or values.dtype != np.float32:
        raise ValueError(f"Existing Qwen embedding artifact has an invalid array: {folder.name}")
    return True


def build_embeddings(*, channels: tuple[str, ...] = CHANNELS,
                     max_lengths: tuple[int, ...] = MAX_LENGTHS,
                     batch_size: int = EMBED_BATCH_SIZE,
                     cache_dir: Path | None = None) -> list[Path]:
    if not channels or set(channels) - set(CHANNELS):
        raise ValueError(f"Unknown content channels: {channels}")
    if not max_lengths or set(max_lengths) - set(MAX_LENGTHS):
        raise ValueError(f"Only fixed max lengths {MAX_LENGTHS} are supported")
    cache_dir = cache_dir or (EDA / "data/processed/modeling/.cache/huggingface")
    app_ids, texts, source_hashes = load_content_texts()
    catalog_sha = hashlib.sha256(app_ids.astype("<i8", copy=False).tobytes()).hexdigest()

    # Do not download/load the model if all requested immutable artifacts exist.
    tasks = []
    for channel in channels:
        for max_length in max_lengths:
            folder = CONTENT_ROOT / _embedding_name(channel, max_length)
            base = _manifest_for_embedding(channel, max_length, source_hashes, 1024)
            base["embedding"]["batch_size"] = batch_size
            base["catalog"]["app_ids_sha256"] = catalog_sha
            config_sha = sha_json(base)
            expected = {**base, "config_sha256": config_sha}
            if _verify_existing_embedding(folder, expected):
                print(f"Verified existing embedding: {folder.name}", flush=True)
                continue
            tasks.append((channel, max_length, folder, expected))
    if not tasks:
        return [CONTENT_ROOT / _embedding_name(c, l) for c in channels for l in max_lengths]

    torch, F, tokenizer, model = _load_qwen(cache_dir)
    dimension = int(model.config.hidden_size)
    if dimension != 1024:
        raise ValueError(f"Pinned model hidden size changed from 1024: {dimension}")
    token_audits = {channel: _tokenization_audit(tokenizer, texts[channel])
                    for channel in {task[0] for task in tasks}}
    for channel, max_length, folder, expected in tasks:
        folder.mkdir(parents=True, exist_ok=True)
        partial_path = folder / "embeddings.partial.npy"
        progress_path = folder / "progress.json"
        progress_signature = expected["config_sha256"]
        if progress_path.is_file() and partial_path.is_file():
            progress = json.loads(progress_path.read_text("utf-8"))
            if progress.get("config_sha256") != progress_signature:
                raise ValueError(f"Partial embedding configuration changed: {folder.name}")
            cursor = int(progress.get("next_row", 0))
            if cursor < 0 or cursor > len(app_ids):
                raise ValueError(f"Invalid partial embedding cursor: {folder.name}")
            vectors = np.load(partial_path, mmap_mode="r+", allow_pickle=False)
            if vectors.shape != (len(app_ids), dimension) or vectors.dtype != np.float32:
                raise ValueError(f"Invalid partial embedding array: {folder.name}")
        else:
            if partial_path.exists() or progress_path.exists():
                raise ValueError(f"Incomplete embedding checkpoint pair: {folder.name}")
            vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                                shape=(len(app_ids), dimension))
            cursor = 0
            write_json(progress_path, {"config_sha256": progress_signature, "next_row": 0})
        all_texts = texts[channel]
        start_time = time.perf_counter()
        print(f"Embedding {channel} max_length={max_length}; resume at row {cursor:,}/37,610", flush=True)
        with torch.inference_mode():
            for begin in range(cursor, len(all_texts), batch_size):
                end = min(begin + batch_size, len(all_texts))
                batch_text = all_texts[begin:end]
                encoded = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=max_length, pad_to_multiple_of=8, add_special_tokens=True)
                empty = np.asarray([not text.strip() for text in batch_text], dtype=bool)
                batch = {key: value.to("cuda", non_blocking=True) for key, value in encoded.items()}
                output = model(**batch, use_cache=False)
                pooled = output.last_hidden_state[:, -1, :].float()
                pooled = F.normalize(pooled, p=2, dim=1)
                rows = pooled.cpu().numpy().astype(np.float32, copy=False)
                rows[empty] = 0.0
                vectors[begin:end] = rows
                vectors.flush()
                cursor = end
                write_json(progress_path, {"config_sha256": progress_signature, "next_row": cursor})
                if begin == 0 or cursor == len(all_texts) or cursor % (batch_size * 250) == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"  {channel}/{max_length}: {cursor:,}/37,610 rows, {elapsed/60:.1f} min", flush=True)
                del encoded, batch, output, pooled, rows
        vectors.flush()
        del vectors
        gc.collect()
        partial_path.replace(folder / "embeddings.npy")
        progress_path.unlink(missing_ok=True)
        manifest = dict(expected)
        manifest["files"] = {"embeddings.npy": {"sha256": digest(folder / "embeddings.npy"),
                                                  "bytes": (folder / "embeddings.npy").stat().st_size}}
        manifest["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0)}
        manifest["tokenization_audit"] = token_audits[channel]
        manifest["elapsed_seconds"] = time.perf_counter() - start_time
        write_json(folder / "manifest.json", manifest)
        print(f"Completed {folder.name}; sha256={manifest['files']['embeddings.npy']['sha256']}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return [CONTENT_ROOT / _embedding_name(c, l) for c in channels for l in max_lengths]


def _load_catalog_and_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Fresh v4 catalog is missing: {CATALOG_PATH}")
    table = pq.read_table(CATALOG_PATH, columns=["app_id", "linux"])
    rows = table.to_pydict()
    app_ids = np.asarray(rows["app_id"], dtype=np.int64)
    linux = np.asarray(rows["linux"], dtype=bool)
    if app_ids.shape != (37_610,) or linux.sum() != 6_630 or np.any(np.diff(app_ids) <= 0):
        raise ValueError("Unexpected frozen v4 catalog dimensions or order")
    if not SUPPORT_PATH.is_file():
        raise FileNotFoundError(f"Fresh v4 positive support is missing: {SUPPORT_PATH}")
    support = np.load(SUPPORT_PATH, allow_pickle=False)
    if support.shape != app_ids.shape or np.any(support < 0):
        raise ValueError("Fresh v4 positive-support vector does not match the catalog")
    return app_ids, linux, support


def _candidate_row_positions(app_ids: np.ndarray, linux: np.ndarray) -> np.ndarray:
    """Return matrix row positions; App IDs must never be used as row indices."""
    app_ids = np.asarray(app_ids, dtype=np.int64)
    linux = np.asarray(linux, dtype=bool)
    if app_ids.ndim != 1 or linux.shape != app_ids.shape:
        raise ValueError("Catalog app IDs and Linux mask must be aligned one-dimensional arrays")
    if np.any(np.diff(app_ids) <= 0):
        raise ValueError("Catalog app IDs must remain ascending")
    return np.flatnonzero(linux).astype(np.int64, copy=False)


def _load_teacher() -> tuple[sparse.csr_matrix, dict[str, Any]]:
    if not TEACHER_PATH.is_file():
        raise FileNotFoundError(
            "V4-5 fitting is waiting for the fresh activity KNN teacher at "
            f"{TEACHER_PATH}; historical teacher matrices are not allowed."
        )
    teacher_manifest_path = TEACHER_PATH.parent / "manifest.json"
    if not teacher_manifest_path.is_file():
        raise FileNotFoundError(f"Fresh teacher manifest is missing: {teacher_manifest_path}")
    teacher_manifest = json.loads(teacher_manifest_path.read_text("utf-8"))
    teacher_sha = digest(TEACHER_PATH)
    if (teacher_manifest.get("status") != "complete" or
            teacher_manifest.get("variant") != "activity_inverse_degree_cosine_k100" or
            teacher_manifest.get("matrix_sha256") != teacher_sha):
        raise ValueError("Fresh v4 activity KNN teacher does not match its completed manifest")
    teacher = sparse.load_npz(TEACHER_PATH).tocsr().astype(np.float32)
    apps, linux, support = _load_catalog_and_support()
    if teacher.shape != (int(linux.sum()), len(apps)):
        raise ValueError(f"Fresh teacher must have shape (Linux candidates, full catalog): {teacher.shape}")
    teacher.sort_indices()
    if np.any(~np.isfinite(teacher.data)) or np.any(teacher.data <= 0):
        raise ValueError("Fresh teacher graph must contain only finite positive similarities")
    return teacher, {"path": TEACHER_PATH.relative_to(EDA).as_posix(),
                     "sha256": teacher_sha, "manifest_sha256": digest(teacher_manifest_path),
                     "shape": list(teacher.shape),
                     "nnz": int(teacher.nnz), "format": "candidate-by-input CSR"}


def _sample_teacher_pairs(teacher: sparse.csr_matrix, candidate_rows: np.ndarray,
                          support: np.ndarray, features: np.ndarray,
                          top_k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    supported_inputs = np.flatnonzero((support > 0) & (np.linalg.norm(features, axis=1) > 0))
    supported_candidate_rows = np.flatnonzero((support[candidate_rows] > 0) &
                                              (np.linalg.norm(features[candidate_rows], axis=1) > 0))
    cand_parts: list[np.ndarray] = []
    input_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    # Keep only the highest teacher neighbors per supported candidate. The
    # teacher's stored top-100 tail is never converted to hard dislikes.
    for cand in supported_candidate_rows:
        start, end = teacher.indptr[cand:cand + 2]
        cols = teacher.indices[start:end]
        vals = teacher.data[start:end]
        eligible = support[cols] > 0
        eligible &= cols != candidate_rows[cand]
        eligible &= np.linalg.norm(features[cols], axis=1) > 0
        if not np.any(eligible):
            continue
        cols, vals = cols[eligible], vals[eligible]
        order = np.lexsort((cols, -vals))[:top_k]
        if order.size:
            cand_parts.append(np.full(order.size, cand, dtype=np.int32))
            input_parts.append(cols[order].astype(np.int32, copy=False))
            value_parts.append(vals[order].astype(np.float32, copy=False))
    if not cand_parts:
        raise ValueError("Teacher contains no supported content pairs for this channel")
    pos_cand = np.concatenate(cand_parts)
    pos_input = np.concatenate(input_parts)
    pos_value = np.concatenate(value_parts)
    rng = np.random.default_rng(seed)
    needed = len(pos_cand) * NEGATIVE_RATIO
    neg_cand_parts: list[np.ndarray] = []
    neg_input_parts: list[np.ndarray] = []
    found = 0
    # Random unlabelled pairs are weakly regularized. Exclude all stored teacher
    # edges, while allowing unobserved pairs to retain uncertainty.
    while found < needed:
        draw = min(max((needed - found) * 2, 4096), 200_000)
        cand = rng.choice(supported_candidate_rows, size=draw, replace=True).astype(np.int32)
        inp = rng.choice(supported_inputs, size=draw, replace=True).astype(np.int32)
        observed = np.asarray(teacher[cand, inp]).reshape(-1) > 0
        observed |= inp == candidate_rows[cand]
        cand, inp = cand[~observed], inp[~observed]
        take = min(len(cand), needed - found)
        if take:
            neg_cand_parts.append(cand[:take])
            neg_input_parts.append(inp[:take])
            found += take
        if len(neg_cand_parts) > 50:
            raise RuntimeError("Could not sample enough weak unlabelled pairs")
    return pos_cand, pos_input, pos_value, np.concatenate(neg_cand_parts), np.concatenate(neg_input_parts)


def _fit_one_alignment(embedding_folder: Path, teacher: sparse.csr_matrix, teacher_info: dict[str, Any],
                       apps: np.ndarray, linux: np.ndarray, support: np.ndarray,
                       dimension: int, top_k: int, device: str = "cuda") -> Path:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    embedding_manifest_path = embedding_folder / "manifest.json"
    embedding_manifest = json.loads(embedding_manifest_path.read_text("utf-8"))
    embedding_path = embedding_folder / "embeddings.npy"
    if embedding_manifest.get("status") != "complete" or digest(embedding_path) != embedding_manifest["files"]["embeddings.npy"]["sha256"]:
        raise ValueError(f"Invalid fixed content embedding: {embedding_folder.name}")
    features_np = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if features_np.shape != (len(apps), 1024) or features_np.dtype != np.float32:
        raise ValueError(f"Embedding row mapping/shape changed: {embedding_folder.name}")
    config = {
        "experiment": "V4-5 two-tower KNN teacher alignment",
        "embedding": embedding_folder.name,
        "embedding_manifest_sha256": digest(embedding_manifest_path),
        "teacher": teacher_info,
        "positive_support_sha256": digest(SUPPORT_PATH),
        "catalog_app_ids_sha256": hashlib.sha256(apps.astype("<i8", copy=False).tobytes()).hexdigest(),
        "supported_games_only": True,
        "input_projector": {"layers": [1024, 256, dimension], "activation": "GELU", "output": "L2 normalized"},
        "candidate_projector": {"layers": [1024, 256, dimension], "activation": "GELU", "output": "L2 normalized"},
        "training_pair_filters": {"both_games_have_positive_support": True,
                                  "both_content_vectors_nonempty": True,
                                  "exclude_candidate_self_pair": True},
        "teacher_positive_sampling": f"highest {top_k} stored neighbors per supported Linux candidate",
        "weak_unlabelled_sampling": {"ratio_to_positive": NEGATIVE_RATIO,
                                     "excludes_all_stored_teacher_edges": True,
                                     "unlabelled_pairs_are_weak_constraints": True,
                                     "loss": "weak upper-margin cosine penalty", "margin": WEAK_NEGATIVE_MARGIN,
                                     "weight": WEAK_NEGATIVE_WEIGHT},
        "positive_loss": "smooth_l1 cosine regression to teacher similarity",
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
        "epochs": ALIGNMENT_EPOCHS,
        "batch_size": ALIGNMENT_BATCH_SIZE,
        "seed": ALIGNMENT_SEED + dimension + top_k,
        "validation_or_confirmation_labels_loaded_for_fitting": False,
        "selection_or_confirmation_metrics_computed": False,
    }
    config_sha = sha_json(config)
    model_name = f"knn_teacher_{embedding_folder.name}_d{dimension}_top{top_k}"
    output = CONTENT_ROOT / model_name
    if (output / "manifest.json").is_file():
        old = json.loads((output / "manifest.json").read_text("utf-8"))
        if old.get("status") == "complete" and old.get("config_sha256") == config_sha:
            for name, file_info in old.get("files", {}).items():
                if digest(output / name) != file_info["sha256"]:
                    raise ValueError(f"Existing alignment output changed: {output / name}")
            print(f"Verified existing alignment: {model_name}", flush=True)
            return output
        raise ValueError(f"Existing alignment output has a different frozen config: {model_name}")

    feature_nonempty = np.linalg.norm(features_np, axis=1) > 0
    features = torch.as_tensor(np.array(features_np, dtype=np.float32, copy=True),
                               dtype=torch.float32, device=device)
    del features_np
    candidate_rows = _candidate_row_positions(apps, linux)
    positive_cand, positive_input, positive_target, negative_cand, negative_input = _sample_teacher_pairs(
        teacher, candidate_rows, support,
        np.asarray(np.load(embedding_path, mmap_mode="r", allow_pickle=False)), top_k,
        int(config["seed"]))
    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, dimension))
        def forward(self, x):
            return F.normalize(self.layers(x), p=2, dim=1)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    input_model, candidate_model = Projector().to(device), Projector().to(device)
    optimizer = torch.optim.AdamW(list(input_model.parameters()) + list(candidate_model.parameters()),
                                  lr=0.001, weight_decay=0.0001)
    p_c = torch.from_numpy(positive_cand.astype(np.int64, copy=False)).to(device)
    p_i = torch.from_numpy(positive_input.astype(np.int64, copy=False)).to(device)
    p_y = torch.from_numpy(positive_target).to(device)
    n_c = torch.from_numpy(negative_cand.astype(np.int64, copy=False)).to(device)
    n_i = torch.from_numpy(negative_input.astype(np.int64, copy=False)).to(device)
    candidate_global = torch.from_numpy(candidate_rows).to(device)
    rng = np.random.default_rng(int(config["seed"]))
    history = []
    started = time.perf_counter()
    print(f"Fitting {model_name}: {len(p_c):,} teacher pairs, {len(n_c):,} weak unlabelled pairs", flush=True)
    for epoch in range(ALIGNMENT_EPOCHS):
        order_p = torch.as_tensor(rng.permutation(len(p_c)), device=device)
        order_n = torch.as_tensor(rng.permutation(len(n_c)), device=device)
        steps = (len(p_c) + ALIGNMENT_BATCH_SIZE - 1) // ALIGNMENT_BATCH_SIZE
        sums = np.zeros(3, dtype=np.float64)
        input_model.train(); candidate_model.train()
        for step in range(steps):
            pb = order_p[step * ALIGNMENT_BATCH_SIZE:(step + 1) * ALIGNMENT_BATCH_SIZE]
            nb = order_n[step * ALIGNMENT_BATCH_SIZE * NEGATIVE_RATIO:
                         (step + 1) * ALIGNMENT_BATCH_SIZE * NEGATIVE_RATIO]
            if len(pb) == 0 or len(nb) == 0:
                continue
            p_input_vectors = input_model(features[p_i[pb]])
            p_candidate_vectors = candidate_model(features[candidate_global[p_c[pb]]])
            pos_score = (p_input_vectors * p_candidate_vectors).sum(dim=1)
            pos_loss = F.smooth_l1_loss(pos_score, p_y[pb])
            n_input_vectors = input_model(features[n_i[nb]])
            n_candidate_vectors = candidate_model(features[candidate_global[n_c[nb]]])
            neg_score = (n_input_vectors * n_candidate_vectors).sum(dim=1)
            neg_loss = F.relu(neg_score - WEAK_NEGATIVE_MARGIN).square().mean()
            loss = pos_loss + WEAK_NEGATIVE_WEIGHT * neg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += (float(loss.detach()), float(pos_loss.detach()), float(neg_loss.detach()))
        history.append({"epoch": epoch + 1, "loss": sums[0] / max(steps, 1),
                        "positive_loss": sums[1] / max(steps, 1),
                        "weak_unlabelled_loss": sums[2] / max(steps, 1)})
        print(f"  epoch {epoch+1}/{ALIGNMENT_EPOCHS} loss={history[-1]['loss']:.5f}", flush=True)

    input_model.eval(); candidate_model.eval()
    input_vectors = np.zeros((len(apps), dimension), dtype=np.float32)
    candidate_vectors = np.zeros((len(apps), dimension), dtype=np.float32)
    nonempty = feature_nonempty
    with torch.inference_mode():
        for begin in range(0, len(apps), 2048):
            end = min(begin + 2048, len(apps))
            input_vectors[begin:end] = input_model(features[begin:end]).cpu().numpy()
            candidate_vectors[begin:end] = candidate_model(features[begin:end]).cpu().numpy()
    input_vectors[~nonempty] = 0.0
    candidate_vectors[~nonempty] = 0.0
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "input_embeddings.npy", input_vectors, allow_pickle=False)
    np.save(output / "candidate_embeddings.npy", candidate_vectors, allow_pickle=False)
    torch.save({"input_projector": input_model.state_dict(), "candidate_projector": candidate_model.state_dict(),
                "input_dimension": 1024, "output_dimension": dimension}, output / "projectors.pt")
    manifest = {"status": "complete", "name": model_name, "config": config, "config_sha256": config_sha,
                "training": {"positive_pairs": int(len(positive_cand)), "weak_unlabelled_pairs": int(len(negative_cand)),
                             "supported_input_games": int(np.count_nonzero(support)),
                             "supported_linux_candidates": int(np.count_nonzero(support[candidate_rows])),
                             "games_with_nonempty_content": int(nonempty.sum()), "epochs": history,
                             "elapsed_seconds": time.perf_counter() - started},
                "artifacts": {"input_vectors": "input_embeddings.npy",
                              "candidate_vectors": "candidate_embeddings.npy",
                              "row_order": "catalog app_ids ascending; candidate vectors use only Linux rows at inference"},
                "fit_data": "Fresh v4 activity KNN graph and full-train positive support only.",
                "ranking_metrics_computed": False}
    manifest["files"] = {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                         for name in ["input_embeddings.npy", "candidate_embeddings.npy", "projectors.pt"]}
    write_json(output / "manifest.json", manifest)
    print(f"Completed {model_name}; elapsed={manifest['training']['elapsed_seconds']/60:.1f} min", flush=True)
    del input_model, candidate_model, optimizer, features, p_c, p_i, p_y, n_c, n_i, candidate_global
    gc.collect(); torch.cuda.empty_cache()
    return output


def train_alignments(*, channels: tuple[str, ...] = CHANNELS,
                     max_lengths: tuple[int, ...] = MAX_LENGTHS,
                     dimensions: tuple[int, ...] = ALIGNED_DIMS,
                     top_ks: tuple[int, ...] = TEACHER_TOP_K) -> list[Path]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("V4-5 alignment requires the prepared CUDA environment")
    apps, linux, support = _load_catalog_and_support()
    content_apps, _, _ = load_content_texts()
    if not np.array_equal(apps, content_apps):
        raise ValueError("Fresh v4 catalog row order differs from frozen v1 content rows")
    teacher, teacher_info = _load_teacher()
    outputs = []
    for channel in channels:
        for max_length in max_lengths:
            embedding_folder = CONTENT_ROOT / _embedding_name(channel, max_length)
            for dimension in dimensions:
                for top_k in top_ks:
                    outputs.append(_fit_one_alignment(embedding_folder, teacher, teacher_info, apps, linux,
                                                      support, dimension, top_k))
    del teacher
    gc.collect(); torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-embeddings", action="store_true", help="Build the fixed Qwen3 V4-6 embeddings")
    parser.add_argument("--train-alignments", action="store_true", help="Fit V4-5 after the fresh KNN teacher exists")
    parser.add_argument("--channel", choices=CHANNELS, action="append", help="Restrict to a channel; repeatable")
    parser.add_argument("--max-length", choices=[str(x) for x in MAX_LENGTHS], action="append",
                        help="Restrict to a fixed token length; repeatable")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory (ignored by git)")
    args = parser.parse_args()
    if not args.build_embeddings and not args.train_alignments:
        args.build_embeddings = True
        args.train_alignments = True
    channels = tuple(args.channel or CHANNELS)
    lengths = tuple(int(x) for x in (args.max_length or [str(x) for x in MAX_LENGTHS]))
    if args.build_embeddings:
        build_embeddings(channels=channels, max_lengths=lengths, batch_size=args.batch_size,
                         cache_dir=args.cache_dir)
    if args.train_alignments:
        train_alignments(channels=channels, max_lengths=lengths)


if __name__ == "__main__":
    main()
