"""Evaluate frozen positive-only baselines on identical saved holdout cohorts.

This is an offline, single-hidden-target review evaluation, not future prediction.
The default cohort is validation; test execution must be explicitly requested.
"""

import argparse
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from eda_large_review_frank import EDA, digest, log
from tfidf_content import DEFAULT_INDEX, TfidfCatalog


DEFAULT_SPLIT = EDA / "data/processed/modeling/split_v1"
DEFAULT_KNN = EDA / "data/processed/modeling/item_knn_v1"
DEFAULT_PROTOCOL = EDA / "data/reference/baseline_protocol_v1.json"
DEFAULT_RUN_LOCK = EDA / "data/reference/baseline_run_v1.json"
PROTOCOL_SHA256 = "62aa2e1029c0fa1d2247d8b79e5e962429d1ec233e5107021efc4e7bf9a30b00"
BUCKETS = ("unseen", "low", "mid_low", "mid_high", "high")


def checked_positions(values, sorted_reference, label):
    values = np.asarray(values)
    positions = np.searchsorted(sorted_reference, values)
    if np.any(positions >= len(sorted_reference)) or np.any(sorted_reference[positions] != values):
        raise ValueError(f"{label} outside frozen catalog/cohort")
    return positions


def pair_matrix(rows, cols, shape, label):
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape).tocsr()
    if matrix.nnz != len(rows) or np.any(matrix.data != 1):
        raise ValueError(f"Duplicate {label} user-game pairs")
    return matrix


def load_cohort(split, cohort, app_ids, candidate_app_ids, protocol):
    manifest = json.loads((split / "manifest.json").read_text("utf-8"))
    if digest(split / "manifest.json") != protocol["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from frozen protocol")
    if manifest["status"] != "complete" or not manifest["validation"]["passed"]:
        raise ValueError("Unverified split")
    tables = {}
    for name in ("users", "query", "targets", "exclusions"):
        relative = f"{cohort}/{name}.parquet"
        if digest(split / relative) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"Split fingerprint changed: {relative}")
        tables[name] = pq.read_table(split / relative).to_pydict()
    users = tables["users"]
    user_ids = np.asarray(users["user_key"], dtype=np.uint64)
    if not len(user_ids) or np.any(user_ids[1:] <= user_ids[:-1]):
        raise ValueError("Cohort users must be unique and sorted")
    expected_count = protocol["cohort_users"][cohort]
    if len(user_ids) != expected_count:
        raise ValueError("Frozen cohort size changed")
    query = tables["query"]
    query_rows = checked_positions(np.asarray(query["user_key"], dtype=np.uint64), user_ids, "Query user")
    query_cols = checked_positions(query["app_id"], app_ids, "Query game")
    linux = np.isin(app_ids[query_cols], candidate_app_ids)
    if not np.array_equal(linux, np.asarray(query["linux"], dtype=bool)):
        raise ValueError("Query Linux flags differ from candidate catalog")
    all_query = pair_matrix(query_rows, query_cols, (len(user_ids), len(app_ids)), "query")
    positive = np.asarray(query["positive"], dtype=bool)
    qpos = pair_matrix(query_rows[positive], query_cols[positive], all_query.shape, "positive query")
    n_total = np.diff(all_query.indptr)
    n_positive = np.diff(qpos.indptr)
    if (np.any(n_total < 5) or np.any(n_positive < 1)
            or not np.array_equal(n_total, users["n_total_inputs"])
            or not np.array_equal(n_positive, users["n_positive_inputs"])):
        raise ValueError("Query counts or baseline eligibility changed")
    exclusions = tables["exclusions"]
    erows = checked_positions(np.asarray(exclusions["user_key"], dtype=np.uint64), user_ids, "Exclusion user")
    ecols = checked_positions(exclusions["app_id"], app_ids, "Exclusion game")
    excluded = pair_matrix(erows, ecols, all_query.shape, "exclusion")
    if (excluded != all_query).nnz:
        raise ValueError("Exclusions must contain every remaining positive and negative query game")
    target = tables["targets"]
    target_rows = checked_positions(np.asarray(target["user_key"], dtype=np.uint64), user_ids, "Target user")
    if not np.array_equal(target_rows, np.arange(len(user_ids))):
        raise ValueError("Exactly one sorted target per cohort user is required")
    target_ids = np.asarray(target["app_id"], dtype=np.int64)
    target_cols = checked_positions(target_ids, app_ids, "Target game")
    target_candidates = checked_positions(target_ids, candidate_app_ids, "Target candidate")
    if (not all(target["positive"]) or not all(target["linux"])
            or np.any(np.asarray(excluded[np.arange(len(user_ids)), target_cols]).ravel())):
        raise ValueError("Targets must be positive Linux games absent from queries and exclusions")
    candidate_cols = checked_positions(candidate_app_ids, app_ids, "Candidate game")
    return {"user_ids": user_ids, "positive": qpos, "n_positive": n_positive,
            "n_total": n_total, "exclusions": excluded[:, candidate_cols].tocsr(),
            "target_ids": target_ids, "target_candidates": target_candidates,
            "hashes": {f"{cohort}/{name}.parquet": manifest["files"][f"{cohort}/{name}.parquet"]["sha256"]
                       for name in tables}}


def rank_positive(scores, app_ids, limit=10):
    """Exact top-k, positive only, with deterministic App ID ascending ties."""
    if not 0 <= limit <= 10:
        raise ValueError("Baseline returns at most 10 recommendations")
    scores = np.asarray(scores)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite personalized scores")
    if limit == 0:
        return np.empty(0, dtype=np.int64)
    valid = np.flatnonzero(scores > 0)
    if len(valid) > limit:
        threshold = np.partition(scores[valid], len(valid) - limit)[-limit]
        valid = valid[scores[valid] >= threshold]
    return valid[np.lexsort((app_ids[valid], -scores[valid]))[:limit]]


def score_batch(model, query, n_positive, kind):
    if kind == "tfidf":
        weights = query.multiply((1.0 / n_positive.astype(np.float32))[:, None]).tocsr()
        profiles = normalize(weights @ model.matrix, norm="l2", copy=False)
        return (profiles @ model.matrix[model.linux].T).toarray()
    return (query @ model.similarities.T).toarray() / n_positive.astype(np.float64)[:, None]


def metrics(ranks, lengths):
    ranks = np.asarray(ranks)
    hit = ranks > 0
    discounts = np.zeros(len(ranks), dtype=np.float64)
    discounts[hit] = 1.0 / np.log2(ranks[hit].astype(np.float64) + 1)
    n = len(ranks)
    return {"users": n, "hits": int(hit.sum()), "Recall@10": float(hit.mean()) if n else None,
            "NDCG@10": float(discounts.mean()) if n else None,
            "empty_recommendation_users": int((lengths == 0).sum()),
            "short_list_users": int((lengths < 10).sum()),
            "mean_recommendations": float(lengths.mean()) if n else None}


def support_buckets(support, thresholds):
    support = np.asarray(support)
    return np.where(support == 0, 0, 1 + np.searchsorted(thresholds, support, side="right"))


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def check_knn_provenance(folder, split, protocol):
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    split_manifest = json.loads((split / "manifest.json").read_text("utf-8"))
    parameters = json.loads((folder / "parameters.json").read_text("utf-8"))
    expected_parameters = {
        "neighbors": protocol["item_knn"]["neighbors"], "minimum_cooccurrence": 1,
        "shrinkage": None, "binary_positive_interactions": True,
        "support_counts_include_positive_singleton_users": True,
        "similarity": "count(i,j)/sqrt(count(i)*count(j)); diagonal forced to zero",
        "neighbor_order": "similarity descending, then App ID ascending",
        "query_score": "sum(candidate similarity to each positive query item)/all positive query count",
        "sparse_index": "candidate-by-input CSR; exact sparse co-occurrence products in candidate blocks",
    }
    if any(parameters.get(key) != value for key, value in expected_parameters.items()) or manifest["parameters"] != parameters:
        raise ValueError("Item-KNN training/scoring settings differ from the frozen baseline")
    provenance = manifest["sources"]
    if (not provenance.get("train_positive_reviews_loaded_by_builder")
            or provenance.get("validation_or_test_reviews_or_ranking_metrics_loaded") is not False):
        raise ValueError("Item-KNN provenance must declare train-only review loading")
    if provenance["split_manifest_sha256"] != protocol["split_manifest_sha256"]:
        raise ValueError("Item-KNN was built from a different split")
    expected_sources = {"train_positive_catalog": "train/positive_catalog.parquet",
                        "input_apps": "catalog/input_apps.parquet", "linux_candidates": "catalog/linux_candidates.parquet"}
    if set(provenance["sources"]) != set(expected_sources):
        raise ValueError("Unexpected Item-KNN training source")
    for source, relative in expected_sources.items():
        actual = provenance["sources"][source]
        expected = split_manifest["files"][relative]
        if (actual["sha256"] != expected["sha256"] or actual["rows"] != expected["rows"]
                or (EDA / actual["file"]).resolve() != (split / relative).resolve()):
            raise ValueError("Item-KNN training/catalog source differs from the frozen split")
    if (not manifest["verification"]["passed"]
            or manifest["support_summary"]["singleton_positive_users_measured"] != 17042189
            or manifest["source_counts"]["train_positive_pairs"] != 78315268
            or manifest["source_counts"]["train_positive_users"] != 29836980):
        raise ValueError("Item-KNN training support or verification differs from the baseline")
    return parameters


def evaluate_model(model, kind, cohort, candidate_ids, support, thresholds, output, batch_size):
    n = len(cohort["user_ids"])
    ranks = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int8)
    target_scores = np.zeros(n, dtype=np.float64)
    recommended_items = set()
    schema = pa.schema([("user_key", pa.uint64()), ("app_id", pa.int64()),
                        ("rank", pa.int8()), ("score", pa.float64())])
    output.mkdir()
    samples = set(np.linspace(0, n - 1, min(12, n), dtype=int).tolist())
    checked_samples = 0
    with pq.ParquetWriter(output / "recommendations.parquet", schema, compression="zstd") as writer:
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            scores = score_batch(model, cohort["positive"][start:stop], cohort["n_positive"][start:stop], kind)
            mask = cohort["exclusions"][start:stop].tocoo()
            scores[mask.row, mask.col] = 0
            target_scores[start:stop] = scores[np.arange(stop - start), cohort["target_candidates"][start:stop]]
            records = []
            for local, row in enumerate(range(start, stop)):
                chosen = rank_positive(scores[local], candidate_ids)
                apps = candidate_ids[chosen]
                lengths[row] = len(chosen)
                matched = np.flatnonzero(apps == cohort["target_ids"][row])
                if len(matched):
                    ranks[row] = int(matched[0] + 1)
                recommended_items.update(apps.tolist())
                if row in samples:
                    input_rows = cohort["positive"][row].indices
                    positive_apps = model.app_ids[input_rows].tolist()
                    excluded_apps = candidate_ids[cohort["exclusions"][row].indices].tolist()
                    scalar = model.recommend(positive_apps, excluded_apps, 10)
                    if [r["app_id"] for r in scalar] != apps.tolist():
                        raise ValueError(f"Batch and single-user rankings differ for {kind}, user row {row}")
                    if scalar:
                        scalar_scores = [r.get("score", r.get("cosine")) for r in scalar]
                        np.testing.assert_allclose(scalar_scores, scores[local, chosen], rtol=3e-5, atol=1e-7)
                    checked_samples += 1
                records.extend({"user_key": int(cohort["user_ids"][row]), "app_id": int(app),
                                "rank": rank, "score": float(scores[local, idx])}
                               for rank, (app, idx) in enumerate(zip(apps, chosen), 1))
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
            if start == 0 or stop == n or (start // batch_size) % 100 == 0:
                log(f"{kind}: {stop:,}/{n:,} users")
    summary = metrics(ranks, lengths)
    target_support = support[cohort["target_candidates"]]
    buckets = support_buckets(target_support, thresholds)
    summary["by_target_train_support"] = {
        label: metrics(ranks[buckets == b], lengths[buckets == b]) for b, label in enumerate(BUCKETS)}
    summary["candidate_catalog_coverage"] = len(recommended_items) / len(candidate_ids)
    summary["distinct_recommended_games"] = len(recommended_items)
    summary["target_positive_personalized_score_users"] = int((target_scores > 0).sum())
    summary["target_zero_personalized_score_users"] = int((target_scores == 0).sum())
    summary["batch_scalar_equivalence_samples"] = checked_samples
    if kind == "tfidf":
        target_supported = np.diff(model.matrix[model.linux].indptr) > 0
        usable_inputs = (np.diff(model.matrix.indptr) > 0).astype(np.int32)
    else:
        target_supported = np.diff(model.similarities.indptr) > 0
        usable_inputs = np.asarray(model.similarities.getnnz(axis=0) > 0, dtype=np.int32)
    summary["unsupported_target_users"] = int((~target_supported[cohort["target_candidates"]]).sum())
    summary["users_without_supported_positive_inputs"] = int((np.asarray(cohort["positive"] @ usable_inputs).ravel() == 0).sum())
    per_user = pa.table({"user_key": pa.array(cohort["user_ids"], type=pa.uint64()),
        "target_app_id": cohort["target_ids"], "n_positive_inputs": cohort["n_positive"],
        "n_total_inputs": cohort["n_total"], "recommendation_count": lengths,
        "target_rank": ranks, "recall_at_10": (ranks > 0).astype(np.float64),
        "target_score": target_scores,
        "ndcg_at_10": np.where(ranks > 0, 1.0 / np.log2(np.maximum(ranks, 1).astype(np.float64) + 1), 0.0),
        "target_train_support": target_support,
        "target_support_bucket": [BUCKETS[b] for b in buckets]})
    pq.write_table(per_user, output / "per_user.parquet", compression="zstd")
    # Read saved predictions independently; don't rely solely on in-memory metrics.
    import duckdb
    con = duckdb.connect()
    try:
        con.register("candidates", pa.table({"app_id": candidate_ids}))
        con.register("users", pa.table({"user_key": cohort["user_ids"]}))
        con.register("excluded", pa.table({"user_key": cohort["user_ids"][cohort["exclusions"].tocoo().row],
                                            "app_id": candidate_ids[cohort["exclusions"].tocoo().col]}))
        from eda_large_review_frank import quote
        con.execute(f"CREATE VIEW recs AS SELECT * FROM read_parquet({quote(output / 'recommendations.parquet')})")
        con.execute(f"CREATE VIEW outcomes AS SELECT * FROM read_parquet({quote(output / 'per_user.parquet')})")
        checks = {
            "nonpositive_or_invalid_scores": "SELECT count(*) FROM recs WHERE NOT isfinite(score) OR score<=0 OR score>1.00001",
            "duplicate_user_game": "SELECT count(*) FROM (SELECT user_key,app_id FROM recs GROUP BY ALL HAVING count(*)<>1)",
            "invalid_or_duplicate_ranks": "SELECT count(*) FROM (SELECT user_key FROM recs GROUP BY user_key HAVING min(rank)<>1 OR max(rank)<>count(*) OR count(DISTINCT rank)<>count(*) OR count(*)>10)",
            "noncandidate_recommendations": "SELECT count(*) FROM recs ANTI JOIN candidates USING(app_id)",
            "unknown_recommended_users": "SELECT count(*) FROM recs ANTI JOIN users USING(user_key)",
            "observed_game_recommendations": "SELECT count(*) FROM recs JOIN excluded USING(user_key,app_id)",
            "missing_outcomes": "SELECT count(*) FROM users ANTI JOIN outcomes USING(user_key)",
            "duplicate_outcomes": "SELECT count(*) FROM (SELECT user_key FROM outcomes GROUP BY user_key HAVING count(*)<>1)",
            "wrong_target_ranks": "SELECT count(*) FROM outcomes o LEFT JOIN recs r ON o.user_key=r.user_key AND o.target_app_id=r.app_id WHERE o.target_rank<>coalesce(r.rank,0)",
            "wrong_list_counts": "SELECT count(*) FROM outcomes o LEFT JOIN (SELECT user_key,count(*) n FROM recs GROUP BY user_key) r USING(user_key) WHERE o.recommendation_count<>coalesce(r.n,0)",
            "wrong_metric_values": "SELECT count(*) FROM outcomes WHERE recall_at_10<>(target_rank>0)::INTEGER OR abs(ndcg_at_10-CASE WHEN target_rank>0 THEN 1/log2(target_rank+1) ELSE 0 END)>1e-12",
            "out_of_order_scores": "SELECT count(*) FROM (SELECT *,lag(score) OVER(PARTITION BY user_key ORDER BY rank) prev_score, lag(app_id) OVER(PARTITION BY user_key ORDER BY rank) prev_app FROM recs) WHERE score>prev_score OR (score=prev_score AND app_id<prev_app)",
        }
        violations = {name: int(con.execute(sql).fetchone()[0]) for name, sql in checks.items()}
        saved = con.execute("SELECT count(*),avg(recall_at_10),avg(ndcg_at_10) FROM outcomes").fetchone()
        if saved[0] != n or not np.allclose(saved[1:], [summary["Recall@10"], summary["NDCG@10"]], rtol=0, atol=1e-12):
            raise ValueError("Saved-file aggregate metrics differ")
        if any(violations.values()):
            raise ValueError(f"Saved prediction invariant violations: {violations}")
        summary["saved_file_violations"] = violations
    finally:
        con.close()
    return summary


def diagnostic_controls(cohort, candidate_ids, support, thresholds):
    """Train-popularity ranking and exact uniform-without-replacement expectations."""
    n = len(cohort["user_ids"])
    ranks = np.zeros(n, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int8)
    ordered = np.lexsort((candidate_ids, -support))
    ordered = ordered[support[ordered] > 0]
    eligible = len(candidate_ids) - np.diff(cohort["exclusions"].indptr)
    random_recall = np.minimum(10, eligible) / eligible
    discounts = 1.0 / np.log2(np.arange(1, 11) + 1)
    random_ndcg = np.array([discounts[:min(10, int(count))].sum() / count for count in eligible])
    for row in range(n):
        excluded = set(cohort["exclusions"][row].indices.tolist())
        chosen = []
        for idx in ordered:
            if idx not in excluded:
                chosen.append(idx)
            if len(chosen) == 10:
                break
        lengths[row] = len(chosen)
        match = np.flatnonzero(np.asarray(chosen) == cohort["target_candidates"][row])
        if len(match):
            ranks[row] = int(match[0] + 1)
    buckets = support_buckets(support[cohort["target_candidates"]], thresholds)
    pop = metrics(ranks, lengths)
    pop["by_target_train_support"] = {label: metrics(ranks[buckets == b], lengths[buckets == b]) for b, label in enumerate(BUCKETS)}
    return {"train_popularity_diagnostic_only_not_personalized_fallback": pop,
            "uniform_random_exact_expectation_not_sampled": {
                "users": n, "Recall@10": float(random_recall.mean()), "NDCG@10": float(random_ndcg.mean()),
                "by_target_train_support": {label: {"users": int((buckets == b).sum()),
                    "Recall@10": float(random_recall[buckets == b].mean()) if np.any(buckets == b) else None,
                    "NDCG@10": float(random_ndcg[buckets == b].mean()) if np.any(buckets == b) else None}
                    for b, label in enumerate(BUCKETS)}}}


def run(args):
    from item_knn import ItemKnnCatalog
    protocol = json.loads(args.protocol.read_text("utf-8"))
    if digest(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("Protocol differs from the a-priori v1 rules; use a new version for changes")
    if protocol["status"] != "frozen_before_ranking_metrics":
        raise ValueError("An a-priori frozen evaluation protocol is required")
    if digest(args.tfidf / "manifest.json") != protocol["tfidf_manifest_sha256"]:
        raise ValueError("TF-IDF index differs from protocol")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    run_lock = json.loads(args.run_lock.read_text("utf-8"))
    frozen_hashes = {"protocol_sha256": digest(args.protocol),
                     "split_manifest_sha256": digest(args.split / "manifest.json"),
                     "tfidf_manifest_sha256": digest(args.tfidf / "manifest.json"),
                     "item_knn_manifest_sha256": digest(args.knn / "manifest.json"),
                     "evaluator_sha256": digest(__file__)}
    if run_lock["status"] != "frozen_before_validation" or any(run_lock[key] != value for key, value in frozen_hashes.items()):
        raise ValueError("Model, evaluator, split or protocol differs from the frozen run lock")
    if (run_lock["tfidf_code_sha256"] != digest(Path(__file__).with_name("tfidf_content.py"))
            or run_lock["item_knn_code_sha256"] != digest(Path(__file__).with_name("item_knn.py"))):
        raise ValueError("A model's retrieval implementation differs from the frozen run lock")
    validation_evidence = None
    if args.cohort == "test":
        if args.validation_result is None:
            raise ValueError("Test evaluation requires --validation-result from this same frozen run")
        validation_evidence = json.loads((args.validation_result / "manifest.json").read_text("utf-8"))
        if (validation_evidence["status"] != "complete" or validation_evidence["cohort"] != "validation"
                or validation_evidence["users"] != protocol["cohort_users"]["validation"]
                or validation_evidence["run_lock_sha256"] != digest(args.run_lock)
                or any(validation_evidence[key] != frozen_hashes[key] for key in frozen_hashes if key != "evaluator_sha256")
                or validation_evidence["code_sha256"]["evaluator"] != frozen_hashes["evaluator_sha256"]):
            raise ValueError("Validation evidence differs from this frozen run or is incomplete")
        for relative, info in validation_evidence["files"].items():
            if digest(args.validation_result / relative) != info["sha256"]:
                raise ValueError("Saved validation results changed")
    tfidf = TfidfCatalog.load(args.tfidf)
    knn = ItemKnnCatalog.load(args.knn)
    if not np.array_equal(knn.app_ids, tfidf.app_ids) or not np.array_equal(knn.candidate_app_ids, tfidf.app_ids[tfidf.linux]):
        raise ValueError("Models must share the exact input/candidate catalog and order")
    # The builder records its own parameters. Require the same untuned K here.
    parameters = check_knn_provenance(args.knn, args.split, protocol)
    if parameters["neighbors"] != protocol["item_knn"]["neighbors"]:
        raise ValueError("Item-KNN K differs from frozen protocol")
    candidate_ids = knn.candidate_app_ids
    if len(knn.app_ids) != protocol["input_games"] or len(candidate_ids) != protocol["linux_candidates"]:
        raise ValueError("Catalog counts differ from frozen protocol")
    support = np.asarray(knn.support)[checked_positions(candidate_ids, knn.app_ids, "Candidate")]
    nonzero_support = support[support > 0]
    thresholds = np.quantile(nonzero_support, [0.25, 0.5, 0.75]).tolist()
    cohort = load_cohort(args.split, args.cohort, knn.app_ids, candidate_ids, protocol)
    stage = args.output.parent / f".{args.output.name}.building-{uuid.uuid4().hex[:10]}"
    stage.mkdir(parents=True)
    result = {"version": "positive-review-baselines-v1", "status": "running", "cohort": args.cohort,
        "started_utc": datetime.now(timezone.utc).isoformat(), "users": len(cohort["user_ids"]),
        "protocol_sha256": digest(args.protocol), "protocol": protocol,
        "run_lock_sha256": digest(args.run_lock),
        "validation_result_sha256": digest(args.validation_result / "manifest.json") if validation_evidence else None,
        "split_manifest_sha256": digest(args.split / "manifest.json"),
        "tfidf_manifest_sha256": digest(args.tfidf / "manifest.json"),
        "item_knn_manifest_sha256": digest(args.knn / "manifest.json"),
        "cohort_file_sha256": cohort["hashes"],
        "train_candidate_support_quartile_thresholds": thresholds,
        "models": {}, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                   "batch_size": args.batch_size},
        "code_sha256": {"evaluator": digest(__file__), "tfidf": digest(Path(__file__).with_name("tfidf_content.py")),
                        "item_knn": digest(Path(__file__).with_name("item_knn.py"))}}
    def save():
        (stage / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    save()
    try:
        for name, model in (("tfidf", tfidf), ("item_knn", knn)):
            result["models"][name] = evaluate_model(model, name, cohort, candidate_ids, support, thresholds,
                                                     stage / name, args.batch_size)
            save()
        result["diagnostic_controls"] = diagnostic_controls(cohort, candidate_ids, support, thresholds)
        result["status"] = "complete"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["files"] = inventory(stage)
        save()
        stage.rename(args.output)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--tfidf", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--knn", type=Path, default=DEFAULT_KNN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-lock", type=Path, default=DEFAULT_RUN_LOCK)
    parser.add_argument("--validation-result", type=Path,
                        help="Completed unchanged validation artifact, required for --cohort test")
    parser.add_argument("--cohort", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "cohort": result["cohort"],
                      "models": result["models"], "controls": result["diagnostic_controls"]}, indent=2))


if __name__ == "__main__":
    main()
