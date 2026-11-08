"""원본(data/raw) → 분석용 parquet(data/processed) 변환.

사용법 (EDA/ 폴더 기준):
    python -I scripts/preprocess.py

원칙
- 원본 파일은 읽기만 한다. 값 보정·필터링은 하지 않고 형식 변환(타입 지정, 중첩 리스트 펼치기)만 수행한다.
- 결측·중복 처리 여부는 노트북에서 판단한다(여기서 행을 지우지 않는다).
- UCSD(B) 파일은 엄격한 JSON이 아니라 줄 단위 Python dict 리터럴이므로 ast.literal_eval로 읽는다.
"""
import ast
import json
from pathlib import Path

import pandas as pd

EDA_DIR = Path(__file__).resolve().parent.parent
RAW = EDA_DIR / "data" / "raw"
OUT = EDA_DIR / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def to_int_id(s):
    """Steam App ID 문자열을 정수로. 숫자가 아니면 <NA> (원본 문자열은 별도 컬럼으로 보존)."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def iter_literal_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ast.literal_eval(line)


# ---------------------------------------------------------------- A
def build_a():
    d = RAW / "A_game_recs"
    rec = pd.read_csv(
        d / "recommendations.csv",
        dtype={"app_id": "int32", "helpful": "int32", "funny": "int32", "hours": "float32",
               "user_id": "int32", "review_id": "int32"},
        true_values=["true"], false_values=["false"],
        parse_dates=["date"],
    )
    rec.to_parquet(OUT / "A_recommendations.parquet", index=False)
    print("A_recommendations", rec.shape)
    del rec

    games = pd.read_csv(d / "games.csv", true_values=["true"], false_values=["false"],
                        parse_dates=["date_release"])
    games.to_parquet(OUT / "A_games.parquet", index=False)
    print("A_games", games.shape)

    users = pd.read_csv(d / "users.csv", dtype={"user_id": "int32", "products": "int32", "reviews": "int32"})
    users.to_parquet(OUT / "A_users.parquet", index=False)
    print("A_users", users.shape)

    meta = pd.read_json(d / "games_metadata.json", lines=True)
    meta.to_parquet(OUT / "A_games_metadata.parquet", index=False)
    print("A_games_metadata", meta.shape)


# ---------------------------------------------------------------- B
def build_b():
    d = RAW / "B_ucsd_steam"

    users, items = [], []
    for line_no, r in enumerate(iter_literal_lines(d / "australian_users_items.json")):
        users.append({"line_no": line_no, "user_id": r["user_id"], "steam_id": r["steam_id"],
                      "items_count": r["items_count"], "n_items_listed": len(r["items"])})
        for it in r["items"]:
            items.append((line_no, r["user_id"], it["item_id"], it["item_name"],
                          it["playtime_forever"], it["playtime_2weeks"]))
    pd.DataFrame(users).to_parquet(OUT / "B_users_items_profiles.parquet", index=False)
    items = pd.DataFrame(items, columns=["line_no", "user_id", "item_id_raw", "item_name",
                                         "playtime_forever", "playtime_2weeks"])
    items["item_id"] = to_int_id(items["item_id_raw"])
    items.to_parquet(OUT / "B_users_items.parquet", index=False)
    print("B_users_items", items.shape, "profiles", len(users))
    del items

    revs = []
    for line_no, r in enumerate(iter_literal_lines(d / "australian_user_reviews.json")):
        for rv in r["reviews"]:
            revs.append({"line_no": line_no, "user_id": r["user_id"], "item_id_raw": rv["item_id"],
                         "recommend": rv["recommend"], "posted": rv["posted"],
                         "last_edited": rv["last_edited"], "helpful": rv["helpful"],
                         "funny": rv["funny"], "review_len": len(rv["review"] or "")})
    revs = pd.DataFrame(revs)
    revs["item_id"] = to_int_id(revs["item_id_raw"])
    # "Posted November 5, 2011." / 연도가 없는 "Posted March 3." (수집 연도 게시물) 두 형식이 있다.
    s = revs["posted"].str.replace(r"^Posted\s+", "", regex=True).str.rstrip(".")
    revs["posted_has_year"] = s.str.contains(r"\d{4}$", regex=True)
    revs["posted_date"] = pd.to_datetime(s.where(revs["posted_has_year"]), format="%B %d, %Y", errors="coerce")
    revs.to_parquet(OUT / "B_reviews.parquet", index=False)
    print("B_reviews", revs.shape)

    games = []
    for r in iter_literal_lines(d / "steam_games.json"):
        games.append({
            "id_raw": r.get("id"), "app_name": r.get("app_name"), "title": r.get("title"),
            "release_date": r.get("release_date"), "genres": r.get("genres"), "tags": r.get("tags"),
            "specs": r.get("specs"), "price": str(r.get("price")) if r.get("price") is not None else None,
            "early_access": r.get("early_access"), "developer": r.get("developer"),
            "publisher": r.get("publisher"), "sentiment": r.get("sentiment"), "url": r.get("url"),
        })
    games = pd.DataFrame(games)
    games["id"] = to_int_id(games["id_raw"])
    games.to_parquet(OUT / "B_steam_games.parquet", index=False)
    print("B_steam_games", games.shape)

    bundles = []
    for r in iter_literal_lines(d / "bundle_data.json"):
        for it in r["items"]:
            bundles.append({"bundle_id": r["bundle_id"], "bundle_name": r["bundle_name"],
                            "item_id_raw": it["item_id"]})
    bundles = pd.DataFrame(bundles)
    bundles["item_id"] = to_int_id(bundles["item_id_raw"])
    bundles.to_parquet(OUT / "B_bundle_items.parquet", index=False)
    print("B_bundle_items", bundles.shape)


# ---------------------------------------------------------------- C
def build_c():
    c = pd.read_csv(RAW / "C_steam_games_2025" / "games_march2025_cleaned.csv", low_memory=False)
    for col in ["windows", "mac", "linux"]:
        c[col] = c[col].map({True: True, False: False, "True": True, "False": False})
    c.to_parquet(OUT / "C_games_2025.parquet", index=False)
    print("C_games_2025", c.shape)


if __name__ == "__main__":
    build_b()
    build_c()
    build_a()
    (OUT / "README.txt").write_text(
        "scripts/preprocess.py 로 생성된 파일. 원본 대비 형식 변환만 수행(행 삭제 없음).\n", "utf-8")
    print(json.dumps(sorted(p.name for p in OUT.glob("*.parquet")), indent=1))
