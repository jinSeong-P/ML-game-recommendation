"""Load either saved baseline and recommend from explicit positive App IDs."""

import argparse
import json
from pathlib import Path

from item_knn import DEFAULT_INDEX as KNN_INDEX, ItemKnnCatalog
from tfidf_content import DEFAULT_INDEX as TFIDF_INDEX, TfidfCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tfidf", "item_knn"], required=True)
    parser.add_argument("--index", type=Path, help="Override the selected model's default artifact folder")
    parser.add_argument("--positive-apps", nargs="+", type=int, required=True)
    parser.add_argument("--exclude-apps", nargs="*", type=int, default=[],
                        help="Every remaining observed game, including negatives; never include a hidden target")
    parser.add_argument("--top-k", type=int, choices=range(11), default=10)
    args = parser.parse_args()
    model = (TfidfCatalog.load(args.index or TFIDF_INDEX) if args.model == "tfidf"
             else ItemKnnCatalog.load(args.index or KNN_INDEX))
    recommendations = model.recommend(args.positive_apps, args.exclude_apps, args.top_k)
    print(json.dumps({"model": args.model, "recommendations": [
        {"app_id": row["app_id"], "score": row.get("score", row.get("cosine"))} for row in recommendations]},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
