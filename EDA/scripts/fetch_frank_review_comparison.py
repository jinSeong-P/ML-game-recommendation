"""Fetch only the pinned Frank reviews identity columns for ForgeMaster EDA."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from fetch_frank_identity_columns import DEFAULT_REVISION, REPO, RangeFile


EDA = Path(__file__).resolve().parents[1]
OUT_DIR = EDA / "data/processed/forgemaster_eda"
OUT_PATH = OUT_DIR / "frank_reviews_identity.parquet"
MANIFEST_PATH = OUT_DIR / "frank_reviews_identity_manifest.json"
EXPECTED_ROWS = 2_631_430
REVIEW_COLUMNS = [
    "author_steamid",
    "appid",
    "recommendationid",
    "voted_up",
    "timestamp_created",
    "timestamp_updated",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{DEFAULT_REVISION}?recursive=true"
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        tree = json.load(response)

    files = {item["path"]: item for item in tree if item["type"] == "file"}
    source_file = files["data/v1/reviews.parquet"]
    source_url = (
        f"https://huggingface.co/datasets/{REPO}/resolve/"
        f"{DEFAULT_REVISION}/{source_file['path']}?download=true"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(source_url, source_file["size"])
    parquet = pq.ParquetFile(remote)
    table = parquet.read(columns=REVIEW_COLUMNS, use_threads=False)

    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} projected rows, found {table.num_rows}")
    if table.column_names != REVIEW_COLUMNS:
        raise ValueError("Projected review columns do not match the requested identity schema")
    author_type = str(table.schema.field("author_steamid").type)
    if author_type not in {"string", "large_string"}:
        raise ValueError(f"Expected string author IDs, found {author_type}")

    staged_path = OUT_PATH.with_name("frank_reviews_identity.staging.parquet")
    pq.write_table(table, staged_path)
    staged_path.replace(OUT_PATH)

    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPO,
        "pinned_commit": DEFAULT_REVISION,
        "scope": "All rows of six review identity columns; review text and all other source columns were not downloaded.",
        "source": {
            "pinned_url": source_url,
            "source_file_bytes": source_file["size"],
            "publisher_sha256_not_independently_checked": source_file.get("lfs", {}).get("oid"),
            "http_bytes_read": remote.bytes_received,
            "http_range_requests": remote.requests,
            "row_groups": parquet.num_row_groups,
        },
        "projection": {
            "path": OUT_PATH.relative_to(EDA).as_posix(),
            "rows": table.num_rows,
            "columns": table.column_names,
            "schema": {field.name: str(field.type) for field in table.schema},
            "local_bytes": OUT_PATH.stat().st_size,
            "sha256": sha256_file(OUT_PATH),
        },
    }
    temporary_manifest = MANIFEST_PATH.with_name("frank_reviews_identity_manifest.staging.json")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(MANIFEST_PATH)

    print(
        f"Frank reviews projection: {table.num_rows:,} rows, "
        f"{len(REVIEW_COLUMNS)} columns, {remote.bytes_received:,} HTTP Range bytes, "
        f"{remote.requests} requests"
    )
    print(f"Projection: {OUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"SHA256: {manifest['projection']['sha256']}")
    print(f"Schema: {json.dumps(manifest['projection']['schema'], sort_keys=True)}")


if __name__ == "__main__":
    main()
