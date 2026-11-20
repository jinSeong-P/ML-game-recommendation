"""Download and fingerprint all 11 CSVs from forgemaster Steam Reviews v1.

Run from the repository root: .venv/Scripts/python EDA/scripts/download_forgemaster.py
Only the dedicated forgemaster_v1 raw directory is written. Existing complete
files are reused; interrupted downloads remain .part files and are retried.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/forgemaster_v1"
REF = "forgemaster/steam-reviews-dataset"
API = "https://www.kaggle.com/api/v1/datasets"
VERSION = 1


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry):
    name, expected = entry["name"], entry["totalBytes"]
    dest = OUT / name
    url = f"{API}/download/{REF}/{urllib.parse.quote(name)}?datasetVersionNumber={VERSION}"
    if not (dest.exists() and dest.stat().st_size == expected):
        for attempt in range(3):
            try:
                print(f"DOWNLOAD {name}: {expected / 1e6:,.1f} MB", flush=True)
                temp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as output:
                    for block in iter(lambda: response.read(8 << 20), b""):
                        output.write(block)
                # Kaggle wraps larger single-file responses in a ZIP archive.
                if zipfile.is_zipfile(temp):
                    unpacked = dest.with_suffix(dest.suffix + ".unpacking")
                    with zipfile.ZipFile(temp) as archive:
                        members = archive.infolist()
                        if len(members) != 1 or members[0].filename != name or members[0].file_size != expected:
                            raise ValueError(f"Unexpected ZIP contents: {name}")
                        with archive.open(members[0]) as source, unpacked.open("wb") as output:
                            for block in iter(lambda: source.read(8 << 20), b""):
                                output.write(block)
                    # Reading the entire member also checks its ZIP CRC.
                    unpacked.replace(temp)
                if temp.stat().st_size != expected:
                    raise ValueError(f"Size mismatch: {name}: {temp.stat().st_size} != {expected}")
                with temp.open("rb") as stream:
                    if not stream.readline().startswith(b"steamid,appid,voted_up,"):
                        raise ValueError(f"Unexpected CSV header: {name}")
                temp.replace(dest)
                break
            except Exception as error:
                if attempt == 2:
                    raise
                print(f"RETRY {name}: {type(error).__name__}: {error}", flush=True)
                time.sleep(2 * (attempt + 1))
    result = {"file": name, "listed_bytes": expected, "local_bytes": dest.stat().st_size,
              "sha256": fingerprint(dest), "url": url}
    print(f"VERIFIED {name}: {result['sha256']}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = get_json(f"{API}/view/{REF}")
    listing = get_json(f"{API}/list/{REF}")
    if metadata["currentVersionNumber"] != VERSION or listing.get("nextPageToken"):
        raise ValueError("Dataset version or file pagination changed; review before downloading")
    (OUT / "kaggle_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), "utf-8")
    (OUT / "kaggle_files.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False), "utf-8")
    entries = listing["datasetFiles"]
    if len(entries) != 11:
        raise ValueError(f"Expected 11 files, got {len(entries)}")
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, entry) for entry in entries]):
            results.append(future.result())
    manifest = {"dataset": REF, "version": VERSION, "license": metadata["licenseName"],
                "source_last_updated": metadata["lastUpdated"],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "All bytes of all 11 listed CSV files, not a sample or projection",
                "total_bytes": sum(item["local_bytes"] for item in results),
                "files": sorted(results, key=lambda item: item["file"])}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"COMPLETE: {len(results)} files; {manifest['total_bytes']:,} bytes", flush=True)


if __name__ == "__main__":
    main()
