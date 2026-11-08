"""Kaggle 공개 데이터셋 파일 단위 다운로드 + manifest 기록.

사용법 (EDA/ 폴더 기준):
    python -I scripts/download_data.py            # 필수 파일 전체
    python -I scripts/download_data.py --only B   # 특정 데이터셋만

Kaggle 공개 데이터셋은 인증 없이
https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}/{file}
로 내려받을 수 있다(2026-11-03 확인). 응답이 zip이면 그대로 풀어 둔다.
원본은 data/raw/ 아래에 보존하고 수정하지 않는다.
"""
import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EDA_DIR = Path(__file__).resolve().parent.parent
RAW = EDA_DIR / "data" / "raw"
API = "https://www.kaggle.com/api/v1"

DATASETS = {
    "A": {
        "ref": "antonkozyriev/game-recommendations-on-steam",
        "dir": "A_game_recs",
        "files": ["games.csv", "games_metadata.json", "users.csv", "recommendations.csv"],
    },
    "B": {
        "ref": "pypiahmad/steam-video-game-and-bundle-data",
        "dir": "B_ucsd_steam",
        # steam_reviews.json/steam_new.json(4.3GB)는 사용자-라벨 구조 확인용으로 앞부분만 받는다(--partial).
        "files": ["australian_user_reviews.json", "australian_users_items.json",
                  "steam_games.json", "bundle_data.json"],
        "partial": {"steam_reviews.json/steam_new.json": 64 * 1024 * 1024},
    },
    "C": {
        "ref": "artermiloff/steam-games-dataset",
        "dir": "C_steam_games_2025",
        "files": ["games_march2025_cleaned.csv"],
    },
}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, max_bytes=None):
    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            n += len(chunk)
            if n % (200 << 20) < (1 << 20):
                print(f"   ... {n / 1e6:,.0f} MB", flush=True)
    tmp.replace(dest)
    return n


def fetch_dataset(key, spec):
    out_dir = RAW / spec["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_json(f"{API}/datasets/view/{spec['ref']}")
    listing = {f["name"]: f["totalBytes"] for f in get_json(f"{API}/datasets/list/{spec['ref']}")["datasetFiles"]}
    entries = []
    targets = [(f, None) for f in spec["files"]] + list(spec.get("partial", {}).items())
    for name, max_bytes in targets:
        local_name = name.replace("/", "__") + (".head" if max_bytes else "")
        dest = out_dir / local_name
        expected = listing.get(name)
        if dest.exists() and (max_bytes or dest.stat().st_size == expected):
            print(f"[{key}] skip (exists) {local_name}")
        else:
            url = f"{API}/datasets/download/{spec['ref']}/{urllib.parse.quote(name, safe='')}"
            print(f"[{key}] downloading {name} ({(expected or 0) / 1e6:,.1f} MB listed)", flush=True)
            download(url, dest, max_bytes)
            # 큰 파일은 zip으로 내려오는 경우가 있다.
            if not max_bytes and zipfile.is_zipfile(dest):
                zpath = dest.with_suffix(dest.suffix + ".zip")
                dest.replace(zpath)
                with zipfile.ZipFile(zpath) as z:
                    member = z.namelist()[0]
                    with z.open(member) as src, open(dest, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                zpath.unlink()
        entries.append({
            "kaggle_file": name,
            "local_path": str(dest.relative_to(EDA_DIR)).replace("\\", "/"),
            "listed_bytes": expected,
            "local_bytes": dest.stat().st_size,
            "partial_head_bytes": max_bytes,
            "sha256": sha256(dest),
        })
    return {
        "key": key,
        "ref": spec["ref"],
        "title": meta.get("titleNullable") or meta.get("title"),
        "license": meta.get("licenseNameNullable") or meta.get("licenseName"),
        "kaggle_version": meta.get("currentVersionNumberNullable") or meta.get("currentVersionNumber"),
        "kaggle_last_updated": meta.get("lastUpdated"),
        "files": entries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(DATASETS))
    args = ap.parse_args()
    keys = args.only or list(DATASETS)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    for k in keys:
        manifest[k] = fetch_dataset(k, DATASETS[k])
        manifest[k]["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    print("manifest ->", manifest_path)


if __name__ == "__main__":
    sys.exit(main())
