"""Independently audit saved Forgemaster ranking metrics and recommendation order.

This read-only audit checks all 64 frozen model/cohort evaluations. It recomputes
per-user Recall@10 and float64 NDCG@10 from hit_rank, compares those values with
the saved per-user rows and published summaries, and checks saved ranking order
and candidate-catalog membership.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


EDA = Path(__file__).resolve().parents[1]
ROOT = EDA / "data/processed/modeling/forgemaster_cap_study_v1"
PROTOCOL = EDA / "data/reference/forgemaster_cap_protocol_v1.json"
CATALOG = EDA / "data/reference/catalog.parquet"
SUMMARY = ROOT / "evaluation_summary.json"
REPORT_JSON = EDA / "reports/forgemaster_model_evaluation.json"
METRICS_CSV = EDA / "reports/tables/forgemaster_models/01_model_metrics.csv"
METRIC_TOLERANCE = 1e-12


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def expected_conditions(protocol: dict) -> list[tuple[str, str]]:
    models = [model["name"] for model in protocol["models"]]
    seeds = protocol["cap_seeds"]
    labels = ["full_history", "capped_full_mask", "capped"]
    conditions = [("full", model) for model in models]
    conditions += [
        (f"cap{seed}_{label}", model)
        for seed in seeds
        for label in labels
        for model in models
    ]
    if len(conditions) != 64 or len(set(conditions)) != 64:
        raise ValueError("Frozen protocol does not define exactly 64 unique conditions")
    return conditions


def index_rows(rows, label: str, expected: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} is missing its results row list")
    indexed = {}
    duplicates = 0
    for row in rows:
        key = (str(row.get("case", "")), str(row.get("model", "")))
        if key in indexed:
            duplicates += 1
        indexed[key] = row
    missing = len(expected - set(indexed))
    unexpected = len(set(indexed) - expected)
    if missing or unexpected or duplicates or len(rows) != 64:
        raise ValueError(
            f"{label} condition coverage failed: expected=64, rows={len(rows)}, "
            f"missing={missing}, unexpected={unexpected}, duplicate_rows={duplicates}"
        )
    return indexed


def check_metric_row(row: dict, metrics: dict[str, float | int], label: str,
                     counters: dict[str, float | int]) -> None:
    for field in ("users", "hits"):
        try:
            raw = float(row[field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} is missing valid {field}") from exc
        if not np.isfinite(raw) or raw != np.floor(raw):
            raise ValueError(f"{label} has invalid {field}")
        actual = int(raw)
        if actual != metrics[field]:
            counters["summary_count_mismatches"] += 1
    for field in ("Recall@10", "NDCG@10"):
        try:
            actual = float(row[field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} is missing valid {field}") from exc
        if not np.isfinite(actual):
            counters["summary_metric_mismatches"] += 1
            continue
        difference = abs(actual - float(metrics[field]))
        counters["max_summary_metric_abs_difference"] = max(
            float(counters["max_summary_metric_abs_difference"]), difference
        )
        if difference > METRIC_TOLERANCE:
            counters["summary_metric_mismatches"] += 1


def audit_recommendation_file(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, int]:
    ordered = con.execute(
        f"""WITH ranked AS (
            SELECT user_key, app_id, rank, score,
                lag(app_id) OVER (PARTITION BY user_key ORDER BY rank, app_id) previous_app_id,
                lag(score) OVER (PARTITION BY user_key ORDER BY rank, app_id) previous_score
            FROM read_parquet({sql_path(path)})
        )
        SELECT
            count(*) FILTER (WHERE NOT isfinite(score)) nonfinite_scores,
            count(*) FILTER (WHERE app_id IS NULL) null_app_ids,
            count(*) FILTER (WHERE previous_score IS NOT NULL AND
                (score > previous_score OR
                 (score = previous_score AND app_id < previous_app_id))) order_errors,
            count(*) FILTER (WHERE candidate.app_id IS NULL) noncandidate_apps,
            count(*) recommendations
        FROM ranked LEFT JOIN audit_candidates candidate USING (app_id)"""
    ).fetchone()
    return {
        "nonfinite_scores": int(ordered[0]),
        "null_app_ids": int(ordered[1]),
        "order_errors": int(ordered[2]),
        "noncandidate_apps": int(ordered[3]),
        "recommendations": int(ordered[4]),
    }


def main() -> None:
    for required in (PROTOCOL, CATALOG, SUMMARY, REPORT_JSON, METRICS_CSV):
        if not required.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {required}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    conditions = expected_conditions(protocol)
    expected = set(conditions)

    summary_doc = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary_doc.get("status") != "complete":
        raise ValueError("evaluation_summary.json is not marked complete")
    summary_rows = index_rows(summary_doc.get("results"), "evaluation_summary.json", expected)

    report_doc = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    report_rows = index_rows(report_doc.get("results"), "public report JSON", expected)

    csv_frame = pd.read_csv(METRICS_CSV)
    required_csv = {"case", "model", "users", "hits", "Recall@10", "NDCG@10"}
    if not required_csv.issubset(csv_frame.columns):
        missing_columns = len(required_csv - set(csv_frame.columns))
        raise ValueError(f"Public metrics CSV is missing {missing_columns} required columns")
    csv_rows = index_rows(
        csv_frame.to_dict(orient="records"), "public metrics CSV", expected
    )

    counters: dict[str, float | int] = {
        "user_recall_mismatches": 0,
        "user_ndcg_mismatches": 0,
        "summary_count_mismatches": 0,
        "summary_metric_mismatches": 0,
        "max_user_recall_abs_difference": 0.0,
        "max_user_ndcg_abs_difference": 0.0,
        "max_summary_metric_abs_difference": 0.0,
        "ranking_order_errors": 0,
        "nonfinite_scores": 0,
        "noncandidate_apps": 0,
        "recommendations": 0,
    }

    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.execute(
        f"CREATE TEMP TABLE audit_candidates AS SELECT app_id "
        f"FROM read_parquet({sql_path(CATALOG)}) WHERE reviewed AND linux"
    )

    try:
        for case, model in conditions:
            folder = ROOT / "evaluations" / case / model
            per_user_path = folder / "per_user.parquet"
            recommendations_path = folder / "recommendations.parquet"
            manifest_path = folder / "manifest.json"
            for path in (per_user_path, recommendations_path, manifest_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Condition artifact missing for {case}/{model}: {path.name}"
                    )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("status") != "complete" or manifest.get("case") != case
                    or manifest.get("model") != model):
                raise ValueError(f"Incomplete or mismatched evaluation manifest: {case}/{model}")

            frame = pd.read_parquet(per_user_path, columns=["hit_rank", "recall", "ndcg"])
            if frame.empty:
                raise ValueError(f"Per-user evaluation is empty: {case}/{model}")

            raw_rank = pd.to_numeric(frame["hit_rank"], errors="coerce").to_numpy(dtype=np.float64)
            if (not np.isfinite(raw_rank).all() or np.any(raw_rank < 0)
                    or np.any(raw_rank > 10) or np.any(raw_rank != np.floor(raw_rank))):
                raise ValueError(f"Invalid hit_rank values in {case}/{model}")
            hit_rank = raw_rank.astype(np.int64)

            expected_recall = (hit_rank > 0).astype(np.float64)
            expected_ndcg = np.zeros(len(hit_rank), dtype=np.float64)
            hit = hit_rank > 0
            expected_ndcg[hit] = 1.0 / np.log2(hit_rank[hit].astype(np.float64) + 1.0)

            stored_recall = pd.to_numeric(frame["recall"], errors="coerce").to_numpy(dtype=np.float64)
            stored_ndcg = pd.to_numeric(frame["ndcg"], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(stored_recall).all() or not np.isfinite(stored_ndcg).all():
                raise ValueError(f"Non-finite per-user metric in {case}/{model}")

            recall_delta = np.abs(stored_recall - expected_recall)
            ndcg_delta = np.abs(stored_ndcg - expected_ndcg)
            max_recall_delta = float(recall_delta.max(initial=0.0))
            max_ndcg_delta = float(ndcg_delta.max(initial=0.0))
            counters["max_user_recall_abs_difference"] = max(
                float(counters["max_user_recall_abs_difference"]), max_recall_delta
            )
            counters["max_user_ndcg_abs_difference"] = max(
                float(counters["max_user_ndcg_abs_difference"]), max_ndcg_delta
            )
            if np.any(recall_delta > METRIC_TOLERANCE):
                counters["user_recall_mismatches"] += 1
            if np.any(ndcg_delta > METRIC_TOLERANCE):
                counters["user_ndcg_mismatches"] += 1

            users = len(hit_rank)
            hits = int(hit.sum())
            metrics = {
                "users": users,
                "hits": hits,
                "Recall@10": hits / users,
                "NDCG@10": float(expected_ndcg.mean(dtype=np.float64)),
            }
            key = (case, model)
            check_metric_row(summary_rows[key], metrics, "evaluation summary", counters)
            check_metric_row(report_rows[key], metrics, "public report JSON", counters)
            check_metric_row(csv_rows[key], metrics, "public metrics CSV", counters)

            ranking = audit_recommendation_file(con, recommendations_path)
            counters["ranking_order_errors"] += ranking["order_errors"]
            counters["nonfinite_scores"] += ranking["nonfinite_scores"] + ranking["null_app_ids"]
            counters["noncandidate_apps"] += ranking["noncandidate_apps"]
            counters["recommendations"] += ranking["recommendations"]
    finally:
        con.close()

    failed = any(
        counters[key] for key in (
            "user_recall_mismatches", "user_ndcg_mismatches",
            "summary_count_mismatches", "summary_metric_mismatches",
            "ranking_order_errors", "nonfinite_scores", "noncandidate_apps",
        )
    )
    if failed:
        raise SystemExit(
            "FAIL: audited=64; "
            f"user_recall_mismatches={counters['user_recall_mismatches']}; "
            f"user_ndcg_mismatches={counters['user_ndcg_mismatches']}; "
            f"summary_count_mismatches={counters['summary_count_mismatches']}; "
            f"summary_metric_mismatches={counters['summary_metric_mismatches']}; "
            f"ranking_order_errors={counters['ranking_order_errors']}; "
            f"nonfinite_or_null={counters['nonfinite_scores']}; "
            f"noncandidate_apps={counters['noncandidate_apps']}; "
            f"max_user_recall_diff={counters['max_user_recall_abs_difference']:.3g}; "
            f"max_user_ndcg_diff={counters['max_user_ndcg_abs_difference']:.3g}; "
            f"max_published_metric_diff={counters['max_summary_metric_abs_difference']:.3g}"
        )

    print(
        "PASS: audited 64/64 model/cohort conditions; "
        f"recommendations={counters['recommendations']}; "
        f"max_user_recall_diff={counters['max_user_recall_abs_difference']:.3g}; "
        f"max_user_ndcg_diff={counters['max_user_ndcg_abs_difference']:.3g}; "
        f"max_published_metric_diff={counters['max_summary_metric_abs_difference']:.3g}; "
        "ranking_order_errors=0; noncandidate_apps=0"
    )


if __name__ == "__main__":
    main()
