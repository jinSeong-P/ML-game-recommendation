#!/usr/bin/env python3
"""Stream the original artermiloff 2024 review archive into canonical Parquet.

The full Kaggle ZIP is retained under EDA/data/raw. ZIP members are parsed in
place, so the uncompressed CSV corpus is never materialized on disk. IDs are
handled as text and malformed scalar values are nulled and counted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "EDA" / "data" / "raw" / "candidate_review" / "artermiloff_2024"
OUT_DIR = ROOT / "EDA" / "data" / "processed" / "candidate_identity" / "artermiloff_2024"
ARCHIVE_PATH = RAW_DIR / "steam-games-reviews-2024-v1.zip"
PARTIAL_ARCHIVE_PATH = RAW_DIR / "steam-games-reviews-2024-v1.zip.part"
DATASET_REF = "artermiloff/steam-games-reviews-2024"
PINNED_VERSION = 1
CARD_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
    f"?datasetVersionNumber={PINNED_VERSION}"
)
USER_AGENT = "curl/8.13.0 (Python urllib; lossless public dataset ingestion)"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PARQUET_ROW_GROUP_ROWS = 100_000
PARQUET_SHARD_ROWS = 1_000_000
PROGRESS_SECONDS = 45

SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=True),
        pa.field("app_id", pa.int64(), nullable=True),
        pa.field("review_id", pa.string(), nullable=True),
        pa.field("positive", pa.bool_(), nullable=True),
        pa.field("timestamp_created", pa.int64(), nullable=True),
        pa.field("timestamp_updated", pa.int64(), nullable=True),
        pa.field("playtime_forever", pa.float64(), nullable=True),
        pa.field("language", pa.string(), nullable=True),
    ]
)
FIELD_INDEX: dict[str, int | None] = {}
SOURCE_FIELDS = {
    "user_id": "author_steamid",
    "review_id": "recommendationid",
    "positive": "voted_up",
    "timestamp_created": "timestamp_created",
    "timestamp_updated": "timestamp_updated",
    "playtime_forever": "author_playtime_forever",
    "language": "language",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ingestion_progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def verify_pinned_card() -> dict[str, Any]:
    card = request_json(CARD_URL)
    live_version = card.get("currentVersionNumber")
    if live_version is not None and int(live_version) < PINNED_VERSION:
        raise RuntimeError(f"Kaggle card has no pinned version {PINNED_VERSION}")
    if card.get("ref") not in (None, DATASET_REF):
        raise RuntimeError("Kaggle returned an unexpected dataset reference")
    return card


def hash_existing(path: Path) -> tuple[hashlib._Hash, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            block = f.read(DOWNLOAD_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest, size


def download_archive(card: dict[str, Any]) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reported_bytes = card.get("totalBytes") or card.get("totalBytesNullable")
    if ARCHIVE_PATH.exists():
        digest, size = hash_existing(ARCHIVE_PATH)
        log(f"Using retained archive ({size:,} bytes); SHA-256 recomputed")
        return {
            "path": str(ARCHIVE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "reported_uncompressed_bytes": reported_bytes,
            "downloaded_from_pinned_endpoint": True,
        }

    offset = PARTIAL_ARCHIVE_PATH.stat().st_size if PARTIAL_ARCHIVE_PATH.exists() else 0
    digest = hashlib.sha256()
    if offset:
        log(f"Resuming retained partial archive at {offset:,} bytes")
        digest, hashed = hash_existing(PARTIAL_ARCHIVE_PATH)
        if hashed != offset:
            raise RuntimeError("Partial archive changed while it was being hashed")

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(DOWNLOAD_URL, headers=headers)
    started = last_report = time.monotonic()
    downloaded = offset
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        content_range = response.headers.get("Content-Range", "")
        if offset and not (status == 206 and content_range.startswith(f"bytes {offset}-")):
            log("Server did not resume at the requested byte; restarting archive download")
            PARTIAL_ARCHIVE_PATH.unlink(missing_ok=True)
            digest = hashlib.sha256()
            offset = downloaded = 0
        mode = "ab" if offset else "wb"
        with PARTIAL_ARCHIVE_PATH.open(mode) as out:
            while True:
                block = response.read(DOWNLOAD_CHUNK_BYTES)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= PROGRESS_SECONDS:
                    elapsed = max(now - started, 0.001)
                    rate = (downloaded - offset) / elapsed / (1024 * 1024)
                    total = response.headers.get("Content-Length")
                    total_hint = f"/{int(total) + offset:,}" if total and total.isdigit() else ""
                    log(f"Download progress: {downloaded:,}{total_hint} bytes ({rate:.1f} MiB/s)")
                    last_report = now

    expected_length = response.headers.get("Content-Length")
    if expected_length and expected_length.isdigit():
        expected_total = offset + int(expected_length)
        if downloaded != expected_total:
            raise RuntimeError(
                f"Archive download ended at {downloaded:,} bytes; expected {expected_total:,}"
            )
    PARTIAL_ARCHIVE_PATH.replace(ARCHIVE_PATH)
    size = ARCHIVE_PATH.stat().st_size
    if reported_bytes is not None and size <= 0:
        raise RuntimeError("Downloaded archive is empty")
    log(f"Archive download complete: {size:,} bytes, SHA-256 {digest.hexdigest()}")
    return {
        "path": str(ARCHIVE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "reported_uncompressed_bytes": reported_bytes,
        "downloaded_from_pinned_endpoint": True,
    }


def crc_verified(handle: Any, expected_crc: int) -> bool:
    """ZipExtFile validates its running CRC when EOF is reached."""
    observed = getattr(handle, "_running_crc", None)
    return observed is not None and observed == expected_crc


def basename_app_id(member_name: str) -> int | None:
    stem = PurePosixPath(member_name).stem
    if not re.fullmatch(r"[0-9]+", stem):
        return None
    value = int(stem)
    return value if 0 <= value <= 2**63 - 1 else None


def token(row: list[str], idx: int | None) -> str | None:
    if idx is None or idx >= len(row):
        return None
    value = row[idx]
    return value if value != "" else None


def parse_id(value: str | None, field: str, invalid: Counter[str]) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"[0-9]+", value) or not any(ch != "0" for ch in value):
        invalid[field] += 1
        return None
    return value


def parse_bool(value: str | None, invalid: Counter[str]) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    invalid["positive"] += 1
    return None


def parse_int(value: str | None, field: str, invalid: Counter[str]) -> int | None:
    if value is None:
        return None
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        invalid[field] += 1
        return None
    try:
        parsed = int(value)
    except ValueError:
        invalid[field] += 1
        return None
    if parsed < -(2**63) or parsed > 2**63 - 1:
        invalid[field] += 1
        return None
    return parsed


def parse_float(value: str | None, invalid: Counter[str]) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        invalid["playtime_forever"] += 1
        return None
    if not math.isfinite(parsed) or parsed < 0:
        invalid["playtime_forever"] += 1
        return None
    return parsed


def map_row(
    row: list[str], app_id: int | None, invalid: Counter[str], nulls: Counter[str]
) -> dict[str, Any]:
    values = {name: token(row, idx) for name, idx in FIELD_INDEX.items()}
    result: dict[str, Any] = {
        "user_id": parse_id(values["user_id"], "user_id", invalid),
        "app_id": app_id,
        "review_id": parse_id(values["review_id"], "review_id", invalid),
        "positive": parse_bool(values["positive"], invalid),
        "timestamp_created": parse_int(values["timestamp_created"], "timestamp_created", invalid),
        "timestamp_updated": parse_int(values["timestamp_updated"], "timestamp_updated", invalid),
        "playtime_forever": parse_float(values["playtime_forever"], invalid),
        "language": (values["language"].strip() or None) if values["language"] is not None else None,
    }
    for name, value in result.items():
        if value is None:
            nulls[name] += 1
    return result


def get_schema_indices(header: list[str]) -> tuple[dict[str, int | None], list[str]]:
    normalized = [column.strip().lower() for column in header]
    index_by_name: dict[str, int] = {}
    for idx, name in enumerate(normalized):
        if name not in index_by_name:
            index_by_name[name] = idx
    indices: dict[str, int | None] = {
        canonical: index_by_name.get(source) for canonical, source in SOURCE_FIELDS.items()
    }
    required = ["author_steamid", "recommendationid", "voted_up"]
    absent_required = [name for name in required if name not in index_by_name]
    return indices, absent_required


def flush_row_group(
    columns: dict[str, list[Any]],
    writer: pq.ParquetWriter | None,
    shard_index: int,
    stage_dir: Path,
) -> tuple[pq.ParquetWriter | None, int, int, int]:
    rows = len(columns["user_id"])
    if rows == 0:
        return writer, shard_index, 0, 0
    if writer is None:
        path = stage_dir / f"reviews_{shard_index:05d}.partial.parquet"
        writer = pq.ParquetWriter(
            str(path),
            SCHEMA,
            compression="zstd",
            compression_level=5,
            use_dictionary=["language"],
            write_statistics=True,
        )
    table = pa.Table.from_pydict(columns, schema=SCHEMA)
    writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_ROWS)
    for name in columns:
        columns[name].clear()
    return writer, shard_index, rows, 1


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def process_archive(
    archive: dict[str, Any], card: dict[str, Any], *, build_parquet: bool = True
) -> dict[str, Any]:
    global FIELD_INDEX
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = OUT_DIR / ".ingest_stage"
    # Refuse to silently append to or replace any prior canonical output.
    existing = list(OUT_DIR.glob("reviews_*.parquet"))
    if existing and build_parquet:
        raise RuntimeError(
            "Canonical reviews_*.parquet output already exists; inspect it before rerunning"
        )
    if build_parquet:
        if stage_dir.exists():
            for item in stage_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    raise RuntimeError(f"Unexpected directory in staging location: {item.name}")
        else:
            stage_dir.mkdir()

    entry_inventory: list[dict[str, Any]] = []
    schema_variants: dict[str, dict[str, Any]] = {}
    schema_counts: Counter[str] = Counter()
    field_invalid: Counter[str] = Counter()
    field_nulls: Counter[str] = Counter()
    malformed_field_counts: Counter[str] = Counter()
    app_ids: set[int] = set()
    csv_entry_count = 0
    non_csv_count = 0
    zip_entry_count = 0
    crc_verified_count = 0
    raw_rows_total = 0
    canonical_rows = 0
    malformed_record_count = 0
    missing_app_mapping_entries = 0
    empty_csv_entries = 0
    parse_error_count = 0
    last_progress = time.monotonic()
    started = last_progress

    columns: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    writer: pq.ParquetWriter | None = None
    shard_index = 0
    rows_in_shard = 0
    with zipfile.ZipFile(ARCHIVE_PATH) as archive_zip:
        infos = archive_zip.infolist()
        zip_entry_count = len(infos)
        csv_infos = [
            info for info in infos
            if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == ".csv"
        ]
        log(f"ZIP inventory found {zip_entry_count:,} entries, including {len(csv_infos):,} CSVs")
        for entry_idx, info in enumerate(infos, 1):
            base = PurePosixPath(info.filename).name
            if info.is_dir():
                continue
            is_csv = PurePosixPath(info.filename).suffix.lower() == ".csv"
            app_id = basename_app_id(info.filename) if is_csv else None
            if is_csv:
                csv_entry_count += 1
                if app_id is None:
                    missing_app_mapping_entries += 1
                else:
                    app_ids.add(app_id)
                entry_info: dict[str, Any] = {
                    "filename": info.filename,
                    "basename": base,
                    "app_id_from_numeric_basename": app_id,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "zip_crc32": f"{info.CRC:08x}",
                    "raw_rows": 0,
                    "canonical_rows": 0,
                    "schema_id": None,
                    "header_columns": [],
                    "missing_required_columns": [],
                    "rows_with_nonstandard_field_count": 0,
                    "csv_parse_errors": 0,
                    "crc_verified": False,
                }
                entry_inventory.append(entry_info)
                if app_id is None:
                    log(f"CSV entry has no valid numeric basename App ID: {info.filename}")
                try:
                    with archive_zip.open(info, "r") as member:
                        text_member = io.TextIOWrapper(member, encoding="utf-8-sig", errors="strict", newline="")
                        reader = csv.reader(text_member, strict=False)
                        try:
                            header = next(reader)
                        except StopIteration:
                            header = []
                        if not header:
                            empty_csv_entries += 1
                            missing_required = ["author_steamid", "recommendationid", "voted_up"]
                            FIELD_INDEX = {}
                        else:
                            FIELD_INDEX, missing_required = get_schema_indices(header)
                            header_normalized = [col.strip().lower() for col in header]
                            schema_id = hashlib.sha256(
                                "\0".join(header_normalized).encode("utf-8")
                            ).hexdigest()[:16]
                            entry_info["schema_id"] = schema_id
                            entry_info["header_columns"] = header
                            variant = schema_variants.setdefault(
                                schema_id,
                                {"columns": header, "normalized_columns": header_normalized, "csv_entries": 0},
                            )
                            variant["csv_entries"] += 1
                            schema_counts[schema_id] += 1
                        entry_info["missing_required_columns"] = missing_required
                        if missing_required:
                            log(
                                f"CSV schema is missing required columns ({len(missing_required)}): "
                                f"{info.filename}"
                            )
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            raw_rows_total += 1
                            if header and len(row) != len(header):
                                malformed_record_count += 1
                                entry_info["rows_with_nonstandard_field_count"] += 1
                                malformed_field_counts[
                                    "short_row" if len(row) < len(header) else "long_row"
                                ] += 1
                            entry = map_row(row, app_id, field_invalid, field_nulls)
                            canonical_rows += 1
                            entry_info["canonical_rows"] += 1
                            if build_parquet:
                                for name in columns:
                                    columns[name].append(entry[name])
                                if len(columns["user_id"]) >= PARQUET_ROW_GROUP_ROWS:
                                    writer, shard_index, just_written, _ = flush_row_group(
                                        columns, writer, shard_index, stage_dir
                                    )
                                    rows_in_shard += just_written
                                    if rows_in_shard >= PARQUET_SHARD_ROWS:
                                        if writer is not None:
                                            writer.close()
                                            writer = None
                                        partial = stage_dir / f"reviews_{shard_index:05d}.partial.parquet"
                                        final = stage_dir / f"reviews_{shard_index:05d}.parquet"
                                        partial.replace(final)
                                        shard_index += 1
                                        rows_in_shard = 0
                            now = time.monotonic()
                            if now - last_progress >= PROGRESS_SECONDS:
                                log(
                                    f"Ingestion progress: {entry_idx:,}/{zip_entry_count:,} ZIP entries; "
                                    f"{raw_rows_total:,} CSV rows; {canonical_rows:,} canonical rows"
                                )
                                last_progress = now
                        entry_info["raw_rows"] = row_count
                        handle = member
                        # ZipExtFile verifies CRC itself while the member is read to EOF.
                        entry_info["crc_verified"] = crc_verified(handle, info.CRC)
                        if not entry_info["crc_verified"]:
                            raise zipfile.BadZipFile("ZIP CSV member CRC was not verified")
                        crc_verified_count += 1
                except (UnicodeDecodeError, csv.Error, zipfile.BadZipFile, OSError) as exc:
                    parse_error_count += 1
                    entry_info["csv_parse_errors"] += 1
                    entry_info["parse_error_type"] = type(exc).__name__
                    # Do not claim completeness or drop this entry silently.
                    log(f"CSV entry processing error in {info.filename}: {type(exc).__name__}")
                    raise
            else:
                non_csv_count += 1
                try:
                    with archive_zip.open(info, "r") as member:
                        while member.read(DOWNLOAD_CHUNK_BYTES):
                            pass
                        if not crc_verified(member, info.CRC):
                            raise zipfile.BadZipFile("ZIP member CRC was not verified")
                        crc_verified_count += 1
                except (zipfile.BadZipFile, OSError):
                    parse_error_count += 1
                    log(f"Non-CSV ZIP entry processing error: {info.filename}")
                    raise
            if entry_idx % 1000 == 0:
                log(
                    f"ZIP progress: {entry_idx:,}/{zip_entry_count:,} entries; "
                    f"{raw_rows_total:,} CSV rows; {crc_verified_count:,} CRC-verified members"
                )

        if build_parquet and columns["user_id"]:
            writer, shard_index, just_written, _ = flush_row_group(
                columns, writer, shard_index, stage_dir
            )
            rows_in_shard += just_written
        if build_parquet and writer is not None:
            writer.close()
            writer = None
            partial = stage_dir / f"reviews_{shard_index:05d}.partial.parquet"
            final = stage_dir / f"reviews_{shard_index:05d}.parquet"
            partial.replace(final)

    parquet_files = (
        sorted(stage_dir.glob("reviews_*.parquet"))
        if build_parquet
        else sorted(OUT_DIR.glob("reviews_*.parquet"))
    )
    parquet_summaries: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.schema_arrow != SCHEMA:
            raise RuntimeError(f"Unexpected canonical schema in {parquet_path.name}")
        parquet_summaries.append(
            {
                "filename": parquet_path.name,
                "rows": parquet_file.metadata.num_rows,
                "row_groups": parquet_file.num_row_groups,
                "bytes": parquet_path.stat().st_size,
            }
        )
        parquet_file.close()
    if sum(item["rows"] for item in parquet_summaries) != canonical_rows:
        raise RuntimeError("Parquet output rows do not match parsed source rows")
    if crc_verified_count != zip_entry_count - sum(1 for info in infos if info.is_dir()):
        raise RuntimeError("One or more archive entries were not CRC verified")

    if build_parquet:
        for parquet_path in parquet_files:
            final = OUT_DIR / parquet_path.name
            parquet_path.replace(final)
        stage_dir.rmdir()
    elapsed = time.monotonic() - started
    result = {
        "dataset_ref": DATASET_REF,
        "version": PINNED_VERSION,
        "version_created": next(
            (v.get("creationDate") for v in card.get("versions", []) if v.get("versionNumber") == PINNED_VERSION),
            None,
        ),
        "license": card.get("licenseName"),
        "archive": archive,
        "ingestion_started_at_utc": datetime.fromtimestamp(
            time.time() - elapsed, timezone.utc
        ).isoformat(timespec="seconds"),
        "ingestion_completed_at_utc": utc_now(),
        "elapsed_seconds": round(elapsed, 2),
        "complete": True,
        "canonical_schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in SCHEMA
        ],
        "source_column_mapping": SOURCE_FIELDS,
        "language_absent_behavior": "canonical language is null when no language source column exists",
        "zip": {
            "entry_count_including_directories": zip_entry_count,
            "non_directory_entry_count": crc_verified_count,
            "csv_entry_count": csv_entry_count,
            "non_csv_file_entry_count": non_csv_count,
            "crc_verified_entry_count": crc_verified_count,
            "all_non_directory_entry_crcs_verified": True,
            "distinct_filename_app_id_count": len(app_ids),
            "filename_entries_without_numeric_app_id": missing_app_mapping_entries,
        },
        "rows": {
            "raw_csv_rows_excluding_headers": raw_rows_total,
            "canonical_rows_written": canonical_rows,
            "rows_with_wrong_field_count": malformed_record_count,
            "wrong_field_count_by_type": dict(malformed_field_counts),
            "csv_parse_errors": parse_error_count,
            "empty_csv_entries": empty_csv_entries,
        },
        "field_quality": {
            "invalid_token_counts_nulled": dict(field_invalid),
            "canonical_null_counts": dict(field_nulls),
            "no_per_user_identifiers_or_token_examples_recorded": True,
        },
        "schema_consistency": {
            "consistent": len(schema_variants) <= 1
            and all(not entry["missing_required_columns"] for entry in entry_inventory),
            "schema_variant_count": len(schema_variants),
            "variants": schema_variants,
        },
        "parquet": {
            "directory": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
            "files": parquet_summaries,
            "shard_count": len(parquet_summaries),
            "total_bytes": sum(item["bytes"] for item in parquet_summaries),
            "row_group_target": PARQUET_ROW_GROUP_ROWS,
            "shard_target_rows": PARQUET_SHARD_ROWS,
            "compression": "zstd",
        },
        "csv_entry_inventory": entry_inventory,
    }
    return result


def write_status(manifest: dict[str, Any] | None, error: str | None = None) -> None:
    status_path = OUT_DIR / "ingestion_status.md"
    if manifest is None:
        body = (
            "# artermiloff 2024 original ingestion\n\n"
            "**Status: incomplete.** Ingestion did not finish successfully.\n\n"
            f"Last recorded error: `{error or 'unknown'}`\n\n"
            f"Raw archive: `{ARCHIVE_PATH.relative_to(ROOT).as_posix()}`\n"
        )
    else:
        zip_info = manifest["zip"]
        row_info = manifest["rows"]
        body = (
            "# artermiloff 2024 original ingestion\n\n"
            f"**Status: {'complete' if manifest['complete'] else 'incomplete'}.**\n\n"
            f"- Kaggle dataset version: {manifest['version']}\n"
            f"- Raw ZIP bytes: {manifest['archive']['size_bytes']:,}\n"
            f"- Raw ZIP SHA-256: `{manifest['archive']['sha256']}`\n"
            f"- CSV archive entries: {zip_info['csv_entry_count']:,}\n"
            f"- Distinct numeric filename App IDs: {zip_info['distinct_filename_app_id_count']:,}\n"
            f"- Raw CSV rows: {row_info['raw_csv_rows_excluding_headers']:,}\n"
            f"- Canonical rows: {row_info['canonical_rows_written']:,}\n"
            f"- Parquet shards: {manifest['parquet']['shard_count']:,}\n"
            f"- Schema variants: {manifest['schema_consistency']['schema_variant_count']:,}\n"
            f"- CRC verified ZIP members: {zip_info['crc_verified_entry_count']:,}/"
            f"{zip_info['non_directory_entry_count']:,}\n"
            f"- Manifest: `ingestion_manifest.json`\n"
        )
    status_path.write_text(body, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log(f"Checking public Kaggle card for pinned v{PINNED_VERSION}")
        card = verify_pinned_card()
        archive = download_archive(card)
        existing_outputs = list(OUT_DIR.glob("reviews_*.parquet"))
        manifest_exists = (OUT_DIR / "ingestion_manifest.json").exists()
        rebuild_manifest_only = bool(existing_outputs) and not manifest_exists
        if rebuild_manifest_only:
            log(
                f"Found {len(existing_outputs):,} completed canonical shards without a manifest; "
                "rebuilding source counts and ZIP CRC audit without rewriting Parquet"
            )
        else:
            log("Starting full ZIP member ingestion and CRC verification")
        manifest = process_archive(archive, card, build_parquet=not rebuild_manifest_only)
        write_manifest(OUT_DIR / "ingestion_manifest.json", manifest)
        write_status(manifest)
        log(
            f"Ingestion complete: {manifest['rows']['canonical_rows_written']:,} rows, "
            f"{manifest['parquet']['shard_count']:,} Parquet shards, "
            f"{manifest['zip']['crc_verified_entry_count']:,} CRC-verified members"
        )
        return 0
    except Exception as exc:
        write_status(None, f"{type(exc).__name__}: {exc}")
        log(f"Ingestion stopped: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    raise SystemExit(main())
