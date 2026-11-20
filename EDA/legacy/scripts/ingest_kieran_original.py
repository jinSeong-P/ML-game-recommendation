#!/usr/bin/env python3
"""Download and ingest the pinned full Kieran Steam reviews source.

The Kaggle version is pinned to 2 (2023-12-10).  Only ``all_reviews.csv`` is
converted to canonical identity/review Parquet; the weighted-score file is
read only to verify its ZIP CRC and is never merged into the main rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "EDA" / "data" / "raw" / "candidate_review" / "kieran_original"
OUTPUT_DIR = ROOT / "EDA" / "data" / "processed" / "candidate_identity" / "kieran_original"
ARCHIVE_NAME = "steam-reviews-v2.zip"
MANIFEST_NAME = "ingestion_manifest.json"

DATASET_OWNER = "kieranpoc"
DATASET_SLUG = "steam-reviews"
DATASET_VERSION_NUMBER = 2
DATASET_VERSION_DATE = "2023-12-10"
DEFAULT_SOURCE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kieranpoc/steam-reviews?datasetVersionNumber=2"
)
EXPECTED_MAIN_CSV_BYTES = 42_487_005_552
EXPECTED_MAIN_ROWS = 113_883_717
EXPECTED_COLUMNS = {
    "user_id": "author_steamid",
    "app_id": "appid",
    "review_id": "recommendationid",
    "positive": "voted_up",
    "timestamp_created": "timestamp_created",
    "timestamp_updated": "timestamp_updated",
    "playtime_forever": "author_playtime_forever",
    "language": "language",
}
SOURCE_COLUMNS = list(dict.fromkeys(EXPECTED_COLUMNS.values()))

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
CSV_BLOCK_BYTES = 16 * 1024 * 1024
SHARD_ROW_LIMIT = 2_500_000
PARQUET_ROW_GROUP_SIZE = 250_000
PROGRESS_INTERVAL_SECONDS = 45

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string()),
        pa.field("app_id", pa.int64()),
        pa.field("review_id", pa.string()),
        pa.field("positive", pa.bool_()),
        pa.field("timestamp_created", pa.int64()),
        pa.field("timestamp_updated", pa.int64()),
        pa.field("playtime_forever", pa.float64()),
        pa.field("language", pa.string()),
    ]
)

INT_PATTERN = r"^[+-]?[0-9]+$"
FLOAT_PATTERN = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"

LOG = logging.getLogger("ingest_kieran_original")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Atomically persist status and aggregate counts without row-level data."""
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def count_mask(mask: pa.Array | pa.ChunkedArray) -> int:
    clean = pc.fill_null(mask, False)
    total = pc.sum(pc.cast(clean, pa.int64())).as_py()
    return int(total or 0)


def prepare_string(raw: pa.Array) -> tuple[pa.Array, pa.Array, pa.Array]:
    """Return trimmed values, a present mask and a missing mask."""
    text = pc.utf8_trim_whitespace(raw)
    empty = pc.fill_null(pc.equal(text, ""), False)
    missing = pc.or_(pc.is_null(text), empty)
    present = pc.invert(missing)
    return text, present, missing


def normalize_int64(raw: pa.Array) -> tuple[pa.Array, int, int]:
    text, present, missing = prepare_string(raw)
    syntax = pc.fill_null(pc.match_substring_regex(text, INT_PATTERN), False)

    negative = pc.fill_null(pc.starts_with(text, "-"), False)
    magnitude = pc.replace_substring_regex(text, r"^[+-]", "")
    magnitude = pc.replace_substring_regex(magnitude, r"^0+", "")
    magnitude = pc.if_else(pc.equal(magnitude, ""), "0", magnitude)

    length = pc.utf8_length(magnitude)
    limit = pc.if_else(
        negative,
        "9223372036854775808",
        "9223372036854775807",
    )
    within_range = pc.or_(
        pc.less(length, 19),
        pc.and_(pc.equal(length, 19), pc.less_equal(magnitude, limit)),
    )
    valid = pc.and_(syntax, pc.fill_null(within_range, False))
    valid = pc.fill_null(valid, False)
    sign = pc.if_else(negative, "-", "")
    normalized = pc.binary_join_element_wise(sign, magnitude, "")
    cast_source = pc.if_else(valid, normalized, pa.scalar(None, pa.string()))
    result = pc.cast(cast_source, pa.int64(), safe=True)

    invalid = pc.and_(present, pc.invert(valid))
    return result, count_mask(invalid), count_mask(missing)


def normalize_float64(raw: pa.Array) -> tuple[pa.Array, int, int]:
    text, present, missing = prepare_string(raw)
    syntax = pc.fill_null(pc.match_substring_regex(text, FLOAT_PATTERN), False)
    length_ok = pc.fill_null(pc.less_equal(pc.utf8_length(text), 128), False)
    candidate = pc.and_(syntax, length_ok)
    candidate_source = pc.if_else(candidate, text, pa.scalar(None, pa.string()))
    parsed = pc.cast(candidate_source, pa.float64(), safe=False)
    finite = pc.fill_null(pc.is_finite(parsed), False)
    valid = pc.and_(candidate, finite)
    result = pc.if_else(valid, parsed, pa.scalar(None, pa.float64()))

    invalid = pc.and_(present, pc.invert(valid))
    return result, count_mask(invalid), count_mask(missing)


def normalize_boolean(raw: pa.Array) -> tuple[pa.Array, int, int]:
    text, present, missing = prepare_string(raw)
    lower = pc.utf8_lower(text)
    true_mask = pc.fill_null(pc.is_in(lower, value_set=pa.array(["1", "true"])), False)
    false_mask = pc.fill_null(pc.is_in(lower, value_set=pa.array(["0", "false"])), False)
    valid = pc.or_(true_mask, false_mask)
    result = pc.if_else(true_mask, True, False)
    result = pc.if_else(valid, result, pa.scalar(None, pa.bool_()))
    invalid = pc.and_(present, pc.invert(valid))
    return result, count_mask(invalid), count_mask(missing)


def normalize_string(raw: pa.Array) -> tuple[pa.Array, int, int]:
    _, present, missing = prepare_string(raw)
    # Preserve nonempty ID and language bytes as strings, including long IDs;
    # never let CSV type inference turn identity fields into floating point.
    result = pc.if_else(present, raw, pa.scalar(None, pa.string()))
    return result, 0, count_mask(missing)


def convert_batch(batch: pa.RecordBatch) -> tuple[pa.RecordBatch, Counter, Counter]:
    source = {name: batch.column(batch.schema.get_field_index(name)) for name in SOURCE_COLUMNS}
    converters = {
        "user_id": (normalize_string, "author_steamid"),
        "app_id": (normalize_int64, "appid"),
        "review_id": (normalize_string, "recommendationid"),
        "positive": (normalize_boolean, "voted_up"),
        "timestamp_created": (normalize_int64, "timestamp_created"),
        "timestamp_updated": (normalize_int64, "timestamp_updated"),
        "playtime_forever": (normalize_float64, "author_playtime_forever"),
        "language": (normalize_string, "language"),
    }
    output_columns = []
    invalid_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for output_name, (converter, source_name) in converters.items():
        column, invalid_count, missing_count = converter(source[source_name])
        output_columns.append(column)
        invalid_counts[output_name] += invalid_count
        missing_counts[output_name] += missing_count
    return (
        pa.RecordBatch.from_arrays(output_columns, schema=OUTPUT_SCHEMA),
        invalid_counts,
        missing_counts,
    )


def download_archive(url: str, archive_path: Path, manifest: dict[str, object], manifest_path: Path) -> None:
    """Stream the original archive to disk while hashing its complete bytes."""
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; kieran-original-ingestor/1.0)"},
    )
    digest = hashlib.sha256()
    downloaded_bytes = 0
    last_log = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=90) as response, part_path.open("wb") as target:
            status = getattr(response, "status", None)
            manifest["download_http_status"] = status
            manifest["download_content_length_bytes"] = response.headers.get("Content-Length")
            write_manifest(manifest_path, manifest)
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                downloaded_bytes += len(chunk)
                now = time.monotonic()
                if now - last_log >= PROGRESS_INTERVAL_SECONDS:
                    LOG.info("Download progress: %.2f GiB", downloaded_bytes / (1024**3))
                    manifest["downloaded_bytes"] = downloaded_bytes
                    write_manifest(manifest_path, manifest)
                    last_log = now
        expected_length = manifest.get("download_content_length_bytes")
        if expected_length and downloaded_bytes != int(expected_length):
            raise IOError(
                f"Downloaded byte count {downloaded_bytes} differs from HTTP Content-Length {expected_length}"
            )
        os.replace(part_path, archive_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    manifest["downloaded_bytes"] = downloaded_bytes
    manifest["archive_sha256"] = digest.hexdigest()
    manifest["archive_path"] = str(archive_path)
    write_manifest(manifest_path, manifest)
    LOG.info("Downloaded original archive: %.2f GiB", downloaded_bytes / (1024**3))


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def scan_member_crc(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    """Read a full ZIP member; ZipExtFile validates CRC at EOF."""
    read_bytes = 0
    last_log = time.monotonic()
    with archive.open(info, "r") as member:
        while True:
            chunk = member.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            read_bytes += len(chunk)
            now = time.monotonic()
            if now - last_log >= PROGRESS_INTERVAL_SECONDS:
                LOG.info("ZIP CRC read progress for %s: %.2f GiB", Path(info.filename).name, read_bytes / (1024**3))
                last_log = now
    if read_bytes != info.file_size:
        raise IOError(
            f"ZIP member {info.filename!r} produced {read_bytes} bytes, expected {info.file_size}"
        )
    crc_records = manifest.setdefault("zip_crc_verified_members", [])
    assert isinstance(crc_records, list)
    crc_records.append(
        {
            "member": info.filename,
            "uncompressed_bytes": read_bytes,
            "zip_crc32": f"{info.CRC:08x}",
            "verified_by_full_read": True,
        }
    )
    write_manifest(manifest_path, manifest)
    LOG.info("ZIP CRC verified: %s", Path(info.filename).name)


def find_main_csv(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    candidates = [
        info
        for info in infos
        if not info.is_dir() and Path(info.filename.replace("\\", "/")).name == "all_reviews.csv"
    ]
    if len(candidates) != 1:
        names = [info.filename for info in candidates]
        raise ValueError(f"Expected exactly one all_reviews.csv ZIP member; found {names}")
    return candidates[0]


def csv_parse_options(structural_errors: list[dict[str, int]]) -> pacsv.ParseOptions:
    def reject_malformed_row(invalid_row: pacsv.InvalidRow) -> str:
        structural_errors.append(
            {
                "row_number": int(invalid_row.number),
                "expected_columns": int(invalid_row.expected_columns),
                "actual_columns": int(invalid_row.actual_columns),
            }
        )
        return "error"

    return pacsv.ParseOptions(
        delimiter=",",
        quote_char='"',
        double_quote=True,
        newlines_in_values=True,
        invalid_row_handler=reject_malformed_row,
    )


def parse_main_csv(
    source: BinaryIO,
    staging_dir: Path,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    """Stream the full CSV to canonical Parquet shards with aggregate errors."""
    structural_errors: list[dict[str, int]] = []
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(
            use_threads=False,
            block_size=CSV_BLOCK_BYTES,
            encoding="utf8",
        ),
        parse_options=csv_parse_options(structural_errors),
        convert_options=pacsv.ConvertOptions(
            column_types={column: pa.string() for column in SOURCE_COLUMNS},
            include_columns=SOURCE_COLUMNS,
            null_values=[""],
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )

    source_rows = 0
    shard_batches: list[pa.RecordBatch] = []
    shard_rows = 0
    shard_index = 0
    invalid_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    started = time.monotonic()
    last_progress = started

    def flush_shard() -> None:
        nonlocal shard_batches, shard_rows, shard_index
        if not shard_batches:
            return
        table = pa.Table.from_batches(shard_batches, schema=OUTPUT_SCHEMA)
        path = staging_dir / f"reviews_{shard_index:05d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            compression_level=3,
            use_dictionary=["language"],
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            write_statistics=True,
        )
        shard_index += 1
        shard_batches = []
        shard_rows = 0

    try:
        while True:
            batch = reader.read_next_batch()
            source_rows += batch.num_rows
            converted, batch_invalid, batch_missing = convert_batch(batch)
            invalid_counts.update(batch_invalid)
            missing_counts.update(batch_missing)
            shard_batches.append(converted)
            shard_rows += converted.num_rows
            if shard_rows >= SHARD_ROW_LIMIT:
                flush_shard()

            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
                elapsed = max(now - started, 1.0)
                LOG.info(
                    "Parse progress: %s rows (%.0f rows/s), %s invalid field tokens",
                    f"{source_rows:,}",
                    source_rows / elapsed,
                    f"{sum(invalid_counts.values()):,}",
                )
                manifest["raw_data_rows_parsed"] = source_rows
                manifest["invalid_token_counts"] = dict(invalid_counts)
                manifest["missing_or_blank_counts"] = dict(missing_counts)
                manifest["parquet_shards_written_to_staging"] = shard_index
                write_manifest(manifest_path, manifest)
                last_progress = now
    except StopIteration:
        pass
    except Exception:
        if structural_errors:
            first_error = structural_errors[0]
            manifest["structural_csv_error_count"] = len(structural_errors)
            manifest["first_structural_csv_error"] = first_error
            write_manifest(manifest_path, manifest)
            raise ValueError(
                "Malformed CSV record encountered; ingestion stopped without dropping it "
                f"(row={first_error['row_number']}, "
                f"expected_columns={first_error['expected_columns']}, "
                f"actual_columns={first_error['actual_columns']})"
            ) from None
        raise

    # Exhausting the CSV stream also reads the ZIP member through EOF, which
    # makes ZipExtFile verify the all_reviews.csv member CRC.
    flush_shard()
    source_rows = int(source_rows)
    if source_rows != EXPECTED_MAIN_ROWS:
        raise ValueError(
            f"Pinned source row-count mismatch: parsed {source_rows:,}, "
            f"dataset card/version expectation is {EXPECTED_MAIN_ROWS:,}"
        )
    manifest["raw_data_rows_parsed"] = source_rows
    manifest["canonical_rows_written"] = source_rows
    manifest["invalid_token_counts"] = dict(invalid_counts)
    manifest["missing_or_blank_counts"] = dict(missing_counts)
    manifest["structural_csv_error_count"] = len(structural_errors)
    manifest["parquet_shards_written_to_staging"] = shard_index
    manifest["parse_elapsed_seconds"] = round(time.monotonic() - started, 2)


def publish_shards(staging_dir: Path, output_dir: Path) -> list[str]:
    """Publish only completed output; failed runs keep data under staging."""
    staged = sorted(staging_dir.glob("reviews_*.parquet"))
    if not staged:
        raise ValueError("Ingestion completed without producing any Parquet shards")
    old_shards = list(output_dir.glob("reviews_*.parquet"))
    for old in old_shards:
        old.unlink()
    published: list[str] = []
    for source in staged:
        destination = output_dir / source.name
        os.replace(source, destination)
        published.append(destination.name)
    staging_dir.rmdir()
    return published


def run(source_url: str, skip_download: bool) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    archive_path = RAW_DIR / ARCHIVE_NAME
    staging_dir = OUTPUT_DIR / ".kieran_original_ingest_staging"
    manifest: dict[str, object] = {
        "source": f"{DATASET_OWNER}/{DATASET_SLUG}",
        "source_version_number": DATASET_VERSION_NUMBER,
        "source_version_date": DATASET_VERSION_DATE,
        "source_url": source_url,
        "source_file_member": "all_reviews/all_reviews.csv",
        "source_file_expected_uncompressed_bytes": EXPECTED_MAIN_CSV_BYTES,
        "source_rows_expected_from_dataset_card": EXPECTED_MAIN_ROWS,
        "canonical_schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in OUTPUT_SCHEMA
        ],
        "download_started_at_utc": utc_now(),
        "status": "running",
        "raw_data_rows_parsed": 0,
        "canonical_rows_written": 0,
        "invalid_token_counts": {},
        "missing_or_blank_counts": {},
        "zip_crc_verified_members": [],
        "weighted_score_above_08_used_in_canonical_rows": False,
    }
    write_manifest(manifest_path, manifest)
    try:
        if not archive_path.exists():
            if skip_download:
                raise FileNotFoundError(f"--skip-download set but archive is missing: {archive_path}")
            download_archive(source_url, archive_path, manifest, manifest_path)
        else:
            LOG.info("Using existing source archive: %s", archive_path.name)

        digest, archive_size = sha256_file(archive_path)
        manifest["archive_path"] = str(archive_path)
        manifest["archive_bytes"] = archive_size
        manifest["archive_sha256"] = digest
        write_manifest(manifest_path, manifest)

        staging_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            main_info = find_main_csv(infos)
            if main_info.file_size != EXPECTED_MAIN_CSV_BYTES:
                raise ValueError(
                    f"Pinned all_reviews.csv size mismatch: {main_info.file_size:,} bytes, "
                    f"expected {EXPECTED_MAIN_CSV_BYTES:,}"
                )
            manifest["zip_members"] = [
                {
                    "member": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "zip_crc32": f"{info.CRC:08x}",
                }
                for info in infos
                if not info.is_dir()
            ]
            manifest["source_file_member"] = main_info.filename
            manifest["source_file_zip_crc32"] = f"{main_info.CRC:08x}"
            write_manifest(manifest_path, manifest)

            # Verify every other archive member, including the weighted subset,
            # by reading it fully. It is intentionally never fed to the parser.
            for info in infos:
                if info.is_dir() or info.filename == main_info.filename:
                    continue
                scan_member_crc(archive, info, manifest, manifest_path)

            with archive.open(main_info, "r") as source:
                parse_main_csv(source, staging_dir, manifest, manifest_path)

            crc_records = manifest.get("zip_crc_verified_members", [])
            assert isinstance(crc_records, list)
            crc_records.append(
                {
                    "member": main_info.filename,
                    "uncompressed_bytes": main_info.file_size,
                    "zip_crc32": f"{main_info.CRC:08x}",
                    "verified_by_full_read": True,
                }
            )
            write_manifest(manifest_path, manifest)

        manifest["published_parquet_shards"] = publish_shards(staging_dir, OUTPUT_DIR)
        manifest["archive_zip_crc_verification"] = "passed_for_all_files"
        manifest["completed_at_utc"] = utc_now()
        manifest["status"] = "complete"
        write_manifest(manifest_path, manifest)
        LOG.info(
            "Completed full ingestion: %s rows across %s Parquet shards",
            f"{manifest['canonical_rows_written']:,}",
            len(manifest["published_parquet_shards"]),
        )
        return 0
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        manifest["failed_at_utc"] = utc_now()
        if staging_dir.exists():
            manifest["staging_shards_retained"] = sorted(
                path.name for path in staging_dir.glob("reviews_*.parquet")
            )
        write_manifest(manifest_path, manifest)
        LOG.error("Ingestion incomplete: %s: %s", type(error).__name__, error)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="Public source endpoint for the pinned Kaggle v2 archive.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an already downloaded archive from the assigned raw directory.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if sys.platform == "win32":
        # Avoid locale-dependent handling of large data paths in console logs.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    return run(args.source_url, args.skip_download)


if __name__ == "__main__":
    raise SystemExit(main())
