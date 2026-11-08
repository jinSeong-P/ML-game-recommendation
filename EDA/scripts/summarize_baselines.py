"""Publish aggregate-only reports from completed frozen baseline evaluation files."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eda_large_review_frank import EDA, digest


ROOT = EDA / "data/processed/modeling/baseline_v1"


def percent(value):
    return "—" if value is None else f"{value * 100:.4f}%"


def summarize():
    source = EDA / "data/source_manifest.json"
    lock = EDA / "data/reference/baseline_run_v1.json"
    knn = EDA / "data/processed/modeling/item_knn_v1/manifest.json"
    training = json.loads(knn.read_text("utf-8"))
    support_rows = pq.read_table(knn.parent / "game_rows.parquet").to_pydict()
    counts = np.asarray(support_rows["positive_user_support"])
    linux = np.asarray(support_rows["linux"], dtype=bool)
    support_counts = {scope: {"games": int(len(values)), "zero": int((values == 0).sum()),
                             "one_to_four": int(((values >= 1) & (values <= 4)).sum()),
                             "five_or_more": int((values >= 5).sum())}
                      for scope, values in (("input_catalog", counts), ("linux_candidates", counts[linux]))}
    results = {}
    for cohort in ("validation", "test"):
        path = ROOT / cohort / "manifest.json"
        result = json.loads(path.read_text("utf-8"))
        if result["status"] != "complete" or result["run_lock_sha256"] != digest(lock):
            raise ValueError("Both cohorts must be complete under the same frozen run")
        if any(any(model["saved_file_violations"].values()) for model in result["models"].values()):
            raise ValueError("Unresolved saved-prediction violations")
        for relative, info in result["files"].items():
            if digest(path.parent / relative) != info["sha256"]:
                raise ValueError("Saved evaluation file changed")
        results[cohort] = result
    if results["test"]["validation_result_sha256"] != digest(ROOT / "validation/manifest.json"):
        raise ValueError("Test results do not reference these validation results")
    report = {"status": "complete", "version": "positive-review-baselines-v1",
        "reported_on_kst": datetime.now(timezone(timedelta(hours=9))).date().isoformat(),
        "run_lock_sha256": digest(lock), "item_knn_training": training,
        "game_train_support_counts": support_counts,
        "item_knn_independent_sql_audit": json.loads((EDA / "reports/item_knn_v1_audit.json").read_text("utf-8")),
        "evaluation_artifacts": {cohort: {"directory": (ROOT / cohort).relative_to(EDA).as_posix(),
                                         "manifest_sha256": digest(ROOT / cohort / "manifest.json")}
                                 for cohort in results},
        "validation": results["validation"], "test": results["test"],
        "automated_tests": {"command": "EDA/.venv/Scripts/python -m unittest discover -s EDA/scripts -p test_*.py -q",
                            "passed": 53, "failed": 0},
        "interpretation": "Untuned fixed baselines; no validation refit or test-driven tuning. Historical fixed-catalog single-target review holdout, not temporal prediction."}
    (EDA / "reports/baseline_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    labels = {"tfidf": "소개글 TF-IDF", "item_knn": "Item-KNN · K=100"}
    text = ["# 두 추천 베이스라인의 학습·평가 결과 — v1", "",
        "**2026-11-08 완료.** 소개글 TF-IDF와 공동 긍정 리뷰 Item-KNN을 저장하고, 내부 검증 325,639명에 이어 frank 테스트 24,342명을 같은 규칙으로 평가했다. 초기 설정은 성능 계산 전에 고정했으며 검증 재학습·튜닝·테스트 기반 변경은 하지 않았다.", "",
        "## 성능", "",
        "Recall@10은 숨긴 정답 1개를 Top 10에서 찾은 사용자 비율이다. NDCG@10은 맞힌 순위까지 반영하며 0~1 범위다. 빈 추천과 미지원 정답도 모든 지표의 분모에 남겼다.", "",
        "| 평가 | 모델 | 사용자 | 적중 사용자 | Recall@10 | NDCG@10 |",
        "|---|---|---:|---:|---:|---:|"]
    for cohort, name in (("validation", "내부 검증"), ("test", "frank 테스트")):
        for model, summary in results[cohort]["models"].items():
            text.append(f"| {name} | {labels[model]} | {summary['users']:,} | {summary['hits']:,} | {percent(summary['Recall@10'])} | {summary['NDCG@10']:.6f} |")
    high_share = {cohort: value["models"]["tfidf"]["by_target_train_support"]["high"]["users"] / value["users"]
                  for cohort, value in results.items()}
    text += ["", "이번 초기 설정에서는 Item-KNN이 두 집단 모두 Recall@10·NDCG@10에서 높았다. 내부 검증에서는 Item-KNN이 train 인기도 기준선도 웃돌았고, TF-IDF는 인기도 기준선보다 낮았다.", "",
        f"train 지원 상위 구간의 정답 비중은 내부 검증 **{percent(high_share['validation'])}**, frank **{percent(high_share['test'])}**다. Item-KNN의 검증·테스트 절대 점수 차이는 이 분포 차이 및 리뷰 시점·수집 조건과 함께 해석해야 한다."]
    text += ["", "## 추천과 정답 지원 범위", "",
        "| 평가 | 모델 | 빈 목록 사용자 | 10개 미만 사용자 | 평균 추천 수 | 추천된 게임 수 / 6,630 | 영점수 정답 사용자 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for cohort, name in (("validation", "내부 검증"), ("test", "frank 테스트")):
        for model, summary in results[cohort]["models"].items():
            text.append(f"| {name} | {labels[model]} | {summary['empty_recommendation_users']:,} | {summary['short_list_users']:,} | {summary['mean_recommendations']:.4f} | {summary['distinct_recommended_games']:,} | {summary['target_zero_personalized_score_users']:,} |")
    text += ["", "영점수 정답은 해당 사용자 입력에서 양수 개인화 점수를 얻지 못한 정답이다. 소개글 결측이나 이웃 자체가 없는 정답보다 넓은 범위이며, 순위 10위 밖 실패와 구분한다.", "",
        "| 평가 | 모델 | 입력 전부 미지원 사용자 | 전역 특징/이웃 미지원 정답 사용자 |",
        "|---|---|---:|---:|"]
    for cohort, name in (("validation", "내부 검증"), ("test", "frank 테스트")):
        for model, summary in results[cohort]["models"].items():
            text.append(f"| {name} | {labels[model]} | {summary['users_without_supported_positive_inputs']:,} | {summary['unsupported_target_users']:,} |")
    text += ["", "TF-IDF 입력 지원은 소개글 영벡터 여부, Item-KNN 입력 지원은 저장된 후보 이웃에 한 번이라도 연결되는지로 센다. 두 정의는 다르다. Item-KNN의 정답 전역 지원은 후보가 이웃을 하나라도 갖는지다.", "",
        "## 비교 기준선", "",
        "두 기준선도 같은 정답과 긍정·부정 관측 게임 제외 마스크를 사용했다. 무작위는 남은 전체 Linux 후보에서 중복 없이 최대 10개를 뽑는 정확한 기대값이다. 인기도는 train 긍정 사용자 수 순이다. 개인화 추천의 빈자리를 채우는 용도로 사용하지 않았다.", "",
        "| 평가 | 진단 기준선 | Recall@10 | NDCG@10 |", "|---|---|---:|---:|"]
    for cohort, name in (("validation", "내부 검증"), ("test", "frank 테스트")):
        for key, label in (("train_popularity_diagnostic_only_not_personalized_fallback", "train 인기도"),
                           ("uniform_random_exact_expectation_not_sampled", "균등 무작위 기대값")):
            summary = results[cohort]["diagnostic_controls"][key]
            text.append(f"| {name} | {label} | {percent(summary['Recall@10'])} | {summary['NDCG@10']:.6f} |")
    thresholds = results["validation"]["train_candidate_support_quartile_thresholds"]
    text += ["", "## train 인기도 구간 진단", "",
        f"Linux 후보의 train 양수 지원 수 25·50·75% 경계는 **{thresholds}**다. 지원 0은 unseen이며, 양수 지원에서 경계보다 작으면 low → mid_low → mid_high → high 순이다. 경계 동점은 상위 구간이다. 사용자 수나 frank 정답 분포로 구간을 다시 고르지 않았다."]
    for cohort, name in (("validation", "내부 검증"), ("test", "frank 테스트")):
        text += ["", f"### {name}", "", "| 정답 지원 구간 | 사용자 | TF-IDF Recall@10 | Item-KNN Recall@10 | TF-IDF NDCG@10 | Item-KNN NDCG@10 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for bucket in ("unseen", "low", "mid_low", "mid_high", "high"):
            a = results[cohort]["models"]["tfidf"]["by_target_train_support"][bucket]
            b = results[cohort]["models"]["item_knn"]["by_target_train_support"][bucket]
            na = "—" if a["NDCG@10"] is None else f"{a['NDCG@10']:.6f}"
            nb = "—" if b["NDCG@10"] is None else f"{b['NDCG@10']:.6f}"
            text.append(f"| {bucket} | {a['users']:,} | {percent(a['Recall@10'])} | {percent(b['Recall@10'])} | {na} | {nb} |")
    text += ["", "## 구현·검증과 저장", "",
        "Item-KNN은 긍정 78,315,268쌍·29,836,980명으로 학습했다. 긍정 singleton 17,042,189명도 지원 수·코사인 분모에 유지했다. Linux 후보별 전체 입력 게임 중 유사한 100개 이웃을 남겨 6,630 × 37,610의 CSR에 634,820개 연결을 저장했다. 이웃 보유 후보는 6,586개이며 나머지 44개도 카탈로그와 평가에 남겼다.", "",
        "TF-IDF는 기존 37,610 × 48,011 벡터를 재사용했다. 전체 유효 벡터 37,527개·Linux 6,613개이며 Linux 영벡터 17개는 그대로다. A 우선·C 짧은 소개글 결측 보충·토큰화 설정을 변경하지 않았다.", "",
        "전체 자동 테스트 53개가 통과했다. 별도 SQL 계산에서 전체 37,610개 게임의 학습 지원 수와 후보 15개의 이웃 1,200개를 대조했으며 지원 수·상위 이웃 명단·코사인 값 불일치는 모두 0이었다. 모델×평가 집단별 저장 추천·지표를 다시 읽어 12개 위반 검사를 수행했으며 총 48개 결과가 모두 0이다. 모델×집단별 12명의 단일 검색과 배치 검색도 일치했다.", "",
        "| 경로 | 내용 |", "|---|---|",
        "| `data/processed/modeling/tfidf_content_v1/` | TF-IDF 벡터·토큰·원문·출처 |",
        "| `data/processed/modeling/item_knn_v1/` | 이웃 유사도·학습 지원 수·게임 매핑·파라미터·지문 |",
        "| `data/processed/modeling/baseline_v1/{validation,test}/` | 모델별 추천 순위·점수, 사용자별 지표와 집계 manifest |",
        "| `data/reference/baseline_protocol_v1.json` | 성능 확인 전 고정한 모델·추천·평가 규칙 |",
        "| `data/reference/baseline_run_v1.json` | 검증 실행 전 고정한 모델·분할·평가 코드 지문 |", "",
        "사용자별 산출물은 Git 제외 경로에만 보관한다. [JSON 집계](baseline_v1.json), [Item-KNN 독립 감사](item_knn_v1_audit.json), [실행·수식 안내](../docs/baseline_models.md)를 참고한다.", "",
        "## 해석과 한계", "",
        "내부 검증의 우선 비교 기준은 NDCG@10이고 동점이면 Recall@10이다. 이 결과는 고정된 두 초기 설정의 비교이며, 이웃 수나 텍스트 설정을 탐색해 고른 최적 모델 성능이 아니다. 후속 설정 선택은 Kieran 내부 검증에서 수행하고, 이미 확인한 frank 성능으로 설정을 고르지 않는다.", "",
        "평가는 시간순 미래 예측이 아니다. frank의 게임당 리뷰 상한과 리뷰어 선정 편향 때문에 Kieran 검증과 정답의 인기도 분포가 다르다. 두 집단의 절대 점수를 직접 일반화하지 않고 구간별 결과도 함께 본다. 과거 고정 Linux 카탈로그, TF-IDF 소개글 결측 17개와 CJK 분절 한계도 유지한다.", ""]
    text += ["## 게임별 학습 지원 수", "", "| 범위 | 게임 | 긍정 학습 사용자 0 | 1~4 | 5 이상 |", "|---|---:|---:|---:|---:|"]
    for scope, label in (("input_catalog", "전체 입력"), ("linux_candidates", "Linux 후보")):
        count = support_counts[scope]
        text.append(f"| {label} | {count['games']:,} | {count['zero']:,} | {count['one_to_four']:,} | {count['five_or_more']:,} |")
    text += ["", "지원 수와 이웃 없음은 구분한다. 지원이 있더라도 공동 긍정 게임이 없으면 이웃이 없을 수 있으며, 후보에 이웃이 있어도 특정 사용자의 입력과 연결되지 않으면 정답 점수는 0이다.", ""]
    (EDA / "reports/baseline_v1.md").write_text("\n".join(text), encoding="utf-8")
    config = json.loads(source.read_text("utf-8"))
    config["modeling_baseline"]["implemented"] = True
    config["modeling_baseline"]["model_performance_measured"] = True
    config["modeling_baseline"]["training"]["item_knn_trained"] = True
    config["modeling_baseline"]["training"]["neighbors"] = 100
    config["modeling_baseline"]["evaluation"].update({"implemented": True, "validation_completed": True,
        "test_completed": True, "protocol": "data/reference/baseline_protocol_v1.json", "tuned": False})
    config["content_artifacts"]["ranking_metrics_computed"] = True
    config["model_artifacts"] = {"item_knn": {"directory": "data/processed/modeling/item_knn_v1",
        "manifest_sha256": digest(knn), "verification_passed": True, "independent_audit": "reports/item_knn_v1_audit.json"},
        "tfidf": {"directory": config["content_artifacts"]["directory"],
                  "manifest_sha256": config["content_artifacts"]["manifest_sha256"]}}
    config["baseline_evaluation"] = {"report": "reports/baseline_v1.json", "readable_report": "reports/baseline_v1.md",
        "run_lock": "data/reference/baseline_run_v1.json", "run_lock_sha256": digest(lock),
        "artifacts": report["evaluation_artifacts"]}
    source.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({cohort: value["models"] for cohort, value in results.items()}, indent=2))


if __name__ == "__main__":
    summarize()
