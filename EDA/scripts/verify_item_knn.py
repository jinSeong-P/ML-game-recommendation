"""Independently audit saved Item-KNN supports and sampled top-neighbor rows in SQL."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from eda_large_review_frank import EDA, digest, quote
from item_knn import DEFAULT_INDEX, ItemKnnCatalog


def verify(folder=DEFAULT_INDEX):
    folder = Path(folder)
    model = ItemKnnCatalog.load(folder)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    train = manifest["sources"]["sources"]["train_positive_catalog"]
    train_path = EDA / train["file"]
    if digest(train_path) != train["sha256"]:
        raise ValueError("Pinned training source changed")
    k = manifest["parameters"]["neighbors"]
    selected = set(np.linspace(0, len(model.candidate_app_ids) - 1, 12, dtype=int).tolist())
    selected.update(np.flatnonzero(np.diff(model.similarities.indptr) == 0)[:3].tolist())
    selected = np.array(sorted(selected), dtype=int)
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=4")
        con.execute(f"CREATE VIEW train AS SELECT user_key,app_id FROM read_parquet({quote(train_path)}) WHERE positive")
        con.execute("CREATE TEMP TABLE support AS SELECT app_id,count(*) n FROM train GROUP BY app_id")
        con.register("saved_support", pa.table({"app_id": model.app_ids, "n": model.support}))
        support_violations = con.execute("""SELECT count(*) FROM saved_support s
            FULL JOIN support t USING(app_id) WHERE coalesce(s.n,0)<>coalesce(t.n,0)""").fetchone()[0]
        con.register("selected", pa.table({"app_id": model.candidate_app_ids[selected]}))
        sql = f"""WITH cooccurrence AS (
            SELECT c.app_id candidate_app_id,t.app_id neighbor_app_id,count(*) n
            FROM (SELECT train.* FROM train JOIN selected USING(app_id)) c
            JOIN train t USING(user_key)
            WHERE c.app_id<>t.app_id
            GROUP BY c.app_id,t.app_id
        ), scored AS (
            SELECT candidate_app_id,neighbor_app_id,
                cooccurrence.n::DOUBLE/sqrt(a.n::DOUBLE*b.n::DOUBLE) similarity
            FROM cooccurrence JOIN support a ON candidate_app_id=a.app_id
            JOIN support b ON neighbor_app_id=b.app_id
        ) SELECT candidate_app_id,neighbor_app_id,similarity FROM scored
            QUALIFY row_number() OVER(PARTITION BY candidate_app_id ORDER BY similarity DESC,neighbor_app_id)<={int(k)}
            ORDER BY candidate_app_id,neighbor_app_id"""
        expected = {(int(a), int(b)): float(s) for a, b, s in con.execute(sql).fetchall()}
        actual = {}
        for row in selected:
            start, end = model.similarities.indptr[row:row+2]
            for col, value in zip(model.similarities.indices[start:end], model.similarities.data[start:end]):
                actual[(int(model.candidate_app_ids[row]), int(model.app_ids[col]))] = float(value)
        key_violations = len(set(actual) ^ set(expected))
        score_violations = sum(not np.isclose(actual[key], expected[key], rtol=0, atol=1e-14)
                               for key in set(actual) & set(expected))
        violations = {"all_item_support_mismatches": int(support_violations),
                      "sampled_top_neighbor_membership_mismatches": int(key_violations),
                      "sampled_cosine_value_mismatches": int(score_violations)}
        if any(violations.values()):
            raise ValueError(f"Item-KNN SQL audit failed: {violations}")
        return {"passed": True, "model_manifest_sha256": digest(folder / "manifest.json"),
                "train_source_sha256": digest(train_path), "support_games_checked": len(model.app_ids),
                "sampled_candidate_rows": len(selected), "sampled_neighbor_edges": len(actual),
                "method": "Independent SQL train counts, user joins, cosine and per-candidate top100; no validation/test reviews or ranking metrics",
                "violations": violations}
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.index)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
