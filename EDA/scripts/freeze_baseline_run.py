"""Freeze model/code fingerprints after training and before validation metrics."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eda_large_review_frank import EDA, digest
from evaluate_baselines import (DEFAULT_KNN, DEFAULT_PROTOCOL, DEFAULT_RUN_LOCK, DEFAULT_SPLIT,
                                PROTOCOL_SHA256, check_knn_provenance)
from item_knn import ItemKnnCatalog
from tfidf_content import DEFAULT_INDEX, TfidfCatalog


def freeze(output=DEFAULT_RUN_LOCK):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen baseline run: {output}")
    if digest(DEFAULT_PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("The v1 protocol changed")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text("utf-8"))
    check_knn_provenance(DEFAULT_KNN, DEFAULT_SPLIT, protocol)
    ItemKnnCatalog.load(DEFAULT_KNN)
    TfidfCatalog.load(DEFAULT_INDEX)
    knn_manifest = json.loads((DEFAULT_KNN / "manifest.json").read_text("utf-8"))
    tfidf_manifest = json.loads((DEFAULT_INDEX / "manifest.json").read_text("utf-8"))
    script_dir = Path(__file__).parent
    if (knn_manifest["code_sha256"]["builder"] != digest(script_dir / "build_item_knn.py")
            or knn_manifest["code_sha256"]["model_and_retrieval"] != digest(script_dir / "item_knn.py")
            or tfidf_manifest["code_sha256"]["text_and_retrieval"] != digest(script_dir / "tfidf_content.py")):
        raise ValueError("The model implementation differs from its training artifact")
    audit = EDA / "reports/item_knn_v1_audit.json"
    checked = json.loads(audit.read_text("utf-8"))
    if not checked["passed"] or checked["model_manifest_sha256"] != digest(DEFAULT_KNN / "manifest.json"):
        raise ValueError("An independent SQL audit of this model must pass before freezing")
    result = {"version": "positive-review-baseline-run-v1", "status": "frozen_before_validation",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(DEFAULT_PROTOCOL), "split_manifest_sha256": digest(DEFAULT_SPLIT / "manifest.json"),
        "tfidf_manifest_sha256": digest(DEFAULT_INDEX / "manifest.json"),
        "item_knn_manifest_sha256": digest(DEFAULT_KNN / "manifest.json"),
        "evaluator_sha256": digest(script_dir / "evaluate_baselines.py"),
        "tfidf_code_sha256": digest(script_dir / "tfidf_content.py"),
        "item_knn_code_sha256": digest(script_dir / "item_knn.py"),
        "item_knn_audit_sha256": digest(audit),
        "ranking_metrics_computed": False}
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_LOCK)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))


if __name__ == "__main__":
    main()
