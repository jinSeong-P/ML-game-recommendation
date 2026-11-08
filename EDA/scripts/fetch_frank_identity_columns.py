"""Read complete identity columns from pinned public Parquets using HTTP Range.

Does not download review text or claim a hash check of unread source bytes.
Downloaded projections are local ignored processed files, not original files.
"""

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

EDA = Path(__file__).resolve().parents[1]
REPO = "frankjc2022/steam-dataset"
DEFAULT_REVISION = "75694cc07a6cdc69f48f9c316ca1e3e38a3f0f89"


def fingerprint(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return {'path':path.relative_to(EDA).as_posix(), 'bytes':path.stat().st_size,
            'sha256':sha.hexdigest()}


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.position = 0
        self.bytes_received = 0
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if count <= 0:
            return b""
        if count > 64 * 1024 * 1024:
            raise ValueError("Unexpected large read; refuse accidental whole-source download")
        start, end = self.position, self.position + count - 1
        # Distinct query keys prevent a proxy reusing a different Range response.
        url = self.url + f"&range_probe={start}-{end}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor exact Range")
            result = response.read(count + 1)
        if len(result) != count:
            raise ValueError("Short/oversized Range response")
        self.position += count
        self.bytes_received += count
        self.requests += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned HF commit by default; pass main explicitly to research a new release")
    args = parser.parse_args()
    out = EDA / "data/processed/frank"
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/revision/{args.revision}", timeout=30) as response:
        metadata = json.load(response)
    commit = metadata["sha"]
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/{commit}?recursive=true", timeout=30) as response:
        tree = json.load(response)
    files = {item["path"]: item for item in tree if item["type"] == "file"}
    projections = {
        "reviews": ["author_steamid", "appid", "recommendationid", "voted_up", "timestamp_created", "timestamp_updated"],
        "user_games": ["steamid", "appid", "playtime_forever"],
    }
    report = {"checked_date": "2026-11-07", "repo": REPO, "commit": commit,
              "scope": "All rows of selected columns; review text and other unselected columns were not downloaded or audited", "files": {}}
    for name, columns in projections.items():
        file = files[f"data/v1/{name}.parquet"]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{file['path']}?download=true"
        remote = RangeFile(url, file["size"])
        parquet = pq.ParquetFile(remote)
        table = parquet.read(columns=columns, use_threads=False)
        user_column = "author_steamid" if name == "reviews" else "steamid"
        if str(table.schema.field(user_column).type) not in ["string", "large_string"]:
            raise ValueError("Identity column is not stored as a string")
        path = out / f"{name}_identity_columns.parquet"
        staged = path.with_suffix(".staging.parquet")
        pq.write_table(table, staged)
        staged.replace(path)
        report["files"][name] = {"source_url": url, "source_bytes": file["size"],
            "publisher_full_file_sha256_not_independently_checked": file.get("lfs", {}).get("oid"),
            "rows": table.num_rows, "row_groups": parquet.num_row_groups, "columns": columns,
            "schema": {field.name: str(field.type) for field in table.schema},
            "http_range_bytes_received": remote.bytes_received, "http_requests": remote.requests,
            "local_projection": fingerprint(path)}
        print(f"{name}: {table.num_rows:,} rows, {remote.bytes_received:,} Range bytes, {remote.requests} requests", flush=True)
    (out / "projection_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
