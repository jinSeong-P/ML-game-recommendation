"""저장된 기준 모델로 사용자가 긍정 평가한 게임에 맞는 게임을 추천합니다."""

import argparse
import json
from pathlib import Path

from item_knn import DEFAULT_INDEX as KNN_INDEX, ItemKnnCatalog
from tfidf_content import DEFAULT_INDEX as TFIDF_INDEX, TfidfCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tfidf", "item_knn"], required=True, help="사용할 추천 모델")
    parser.add_argument("--index", type=Path, help="모델 파일 폴더. 생략하면 해당 모델의 기본 경로 사용")
    parser.add_argument("--positive-apps", nargs="+", type=int, required=True, help="긍정 평가한 게임의 Steam App ID")
    parser.add_argument("--exclude-apps", nargs="*", type=int, default=[],
                        help="추천에서 제외할 이미 리뷰한 게임. 부정 게임을 포함하고 숨긴 정답은 제외")
    parser.add_argument("--top-k", type=int, choices=range(11), default=10, help="추천 개수 (기본 10개)")
    args = parser.parse_args()
    model = (TfidfCatalog.load(args.index or TFIDF_INDEX) if args.model == "tfidf"
             else ItemKnnCatalog.load(args.index or KNN_INDEX))
    recommendations = model.recommend(args.positive_apps, args.exclude_apps, args.top_k)
    print(json.dumps({"model": args.model, "recommendations": [
        {"app_id": row["app_id"], "score": row.get("score", row.get("cosine"))} for row in recommendations]},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
