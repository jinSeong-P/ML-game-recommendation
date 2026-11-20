"""Build the Korean EDA report and an executed, self-contained review notebook."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

EDA=Path(__file__).resolve().parents[1]
REPORT=EDA / "reports/forgemaster_eda.md"
METRICS=EDA / "reports/forgemaster_eda_metrics.json"
TABLES=EDA / "reports/tables/forgemaster"
QUALITY_LABELS={
    "invalid_steamid_rows":"Steam 계정 ID 형식 오류",
    "invalid_appid_rows":"게임 App ID 형식 오류",
    "missing_label_rows":"긍정·부정 라벨 결측",
    "negative_playtime_rows":"플레이 시간이 음수",
    "playtime_at_review_gt_forever":"리뷰 작성 당시 플레이 시간이 수집 시점 누적값보다 큼",
    "updated_before_created":"갱신 시각이 작성 시각보다 빠름",
    "created_outside_publish_window":"작성 시각이 배포 시점 범위 밖",
    "negative_user_metadata":"사용자 메타데이터가 음수",
    "negative_votes":"리뷰 투표 수가 음수",
    "invalid_helpfulness_score":"리뷰 유용성 점수가 유효 범위 밖",
    "non_ascii_text_rows":"ASCII 이외 문자가 포함된 리뷰",
}


def fmt(value):
    return f"{value:,.0f}"


def pct(value):
    return f"{value*100:.2f}%"


def md_table(headers,rows):
    lines=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]
    lines.extend("| "+" | ".join(str(cell).replace("|","/") for cell in row)+" |" for row in rows)
    return "\n".join(lines)


def main():
    m=json.loads(METRICS.read_text("utf-8"))
    raw,clean,quality,dup,coverage=m["raw"],m["clean"],m["quality"],m["duplicates"],m["catalog_coverage"]
    full=m["eligibility"][2]["eligible_users"]
    caps=m["cap_sensitivity"]
    cap_users=np.array([row["eligible_users"] for row in caps])
    gameq=m["reviews_per_game"]["quantiles"]
    userq=m["reviews_per_user"]["quantiles"]
    histories=m["user_history_completeness_proxy"]
    frank=m["frank_comparison"]
    overlap=frank["overlap"]
    elig_overlap=frank["source_eligible_overlap"]
    qflags=[(key,value) for key,value in quality.items() if not key.endswith("_unix")]
    quality_table=md_table(["검사","행 수"],[(QUALITY_LABELS.get(key,key),fmt(value)) for key,value in qflags])
    cap_table=md_table(["난수 시드","남은 리뷰 쌍","평가 가능한 사용자","제한 전 대비 평가 사용자 비율","긍정 비율","100개 초과 게임의 긍정률 평균 절대오차"],
        [(row["seed"],fmt(row["rows"]),fmt(row["eligible_users"]),pct(row["eligible_users"]/full),pct(row["positive_rate"]),
          f"{row['positive_rate_error_games_gt100']['mean_absolute_error']*100:.2f}%p") for row in caps])
    elig_table=md_table(["정답을 숨긴 뒤 최소 리뷰 게임 수","리뷰 수 조건과 Linux 정답 조건 만족","긍정 입력 1개 이상 조건도 만족","선택 가능한 사용자–정답 쌍"],
        [(row["minimum_inputs_after_holdout"],fmt(row["potential_users"]),fmt(row["eligible_users"]),fmt(row["possible_target_pairs"])) for row in m["eligibility"]])
    games=pd.read_csv(TABLES / "03_game_support.csv").head(10)
    top_table=md_table(["App ID","게임","리뷰 쌍","긍정 비율","지정한 Linux 추천 후보"],
        [(int(row.app_id),row.title if pd.notna(row.title) else "고정 게임명 자료 없음",fmt(row.reviews),pct(row.positive_rate),"예" if row.in_linux_candidates else "아니오") for row in games.itertuples()])
    dimensions=md_table(["항목","Forgemaster 전체 데이터","frank 전체 데이터"],[
        ("정리 후 리뷰 쌍",fmt(clean["rows"]),fmt(frank["profile"]["rows"])),
        ("원 Steam 계정",fmt(clean["users"]),fmt(frank["profile"]["users"])),
        ("게임",fmt(clean["apps"]),fmt(frank["profile"]["apps"])),
        ("게임당 최대 리뷰",fmt(m['cap_shape']['maximum']),fmt(frank['raw_per_game_counts']['maximum'])),
        ("긍정 비율",pct(clean["positive_rate"]),pct(frank["profile"]["positive_rate"])),
        ("같은 기본 조건의 평가 사용자",fmt(full),fmt(frank["eligibility_after_existing_quarantine"][2]["eligible_users"]))])
    csv_scan=m["independent_python_csv_validation"]
    csv_strict_failures=sum(row["strict_csv_error"] is not None for row in csv_scan["files"])
    time_audit=m['additional_saved_output_audit']['timestamps']
    followup_report=followup_notebook=''
    if (EDA / 'reports/forgemaster_model_evaluation.md').exists():
        followup='**후속 모델 평가 완료:** Kieran 원본을 복원하고 평가 계정을 학습에서 제외해 다시 학습했다. 결과는 [모델 평가 보고서]({})에서 확인할 수 있다. 이 문서의 모델 파일 부재와 학습·평가 계정 중복 미검사는 그 이전 EDA 단계의 상태다.'
        followup_report=followup.format('forgemaster_model_evaluation.md')
        followup_notebook='\n\n'+followup.format('../reports/forgemaster_model_evaluation.md')
    report=f"""# Forgemaster Steam Reviews 데이터 분석

검토일: 2026-11-08 (KST). [Kaggle 원본](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset), **버전 1**, 2021-04-27 배포, CC0. 기존 Kieran·frank 실험의 데이터 분할과 모델 평가 수치는 보존했다.

{followup_report}

## 분석 결론

**게임당 리뷰를 최대 100개로 줄였을 때의 영향을 확인할 수 있는 데이터다.** 학습·평가 계정 중복을 확인하기 전의 분석이며, 이후 계정 분리와 모델 평가 결과는 위 후속 보고서에 정리했다.

- CSV 11개 전체, **{fmt(m['source_manifest']['total_bytes'])}바이트(5.318GB)**를 내려받아 파일별 크기와 SHA256을 확인했다. Forgemaster는 모든 행과 열을 검사했다. frank는 지정한 버전에서 비교에 필요한 6개 열의 모든 행을 읽었다.
- 원본 **{fmt(raw['rows'])}리뷰·{fmt(raw['users'])}계정·{fmt(raw['apps'])}게임**이다. {fmt(m['cap_shape']['games_gt_100'])}개 게임에 100개 초과 리뷰가 있고, 최다는 **{fmt(m['cap_shape']['maximum'])}개**다. 모든 게임에 일률적인 100개 상한이 걸린 데이터는 아니다.
- 현재 입력·정답 조건을 만족하는 사용자는 **{fmt(full)}명**이다. 같은 데이터에 게임별 무작위 최대 100개 제한을 적용하면 난수 시드 5개로 만든 표본에서 **{fmt(cap_users.min())}–{fmt(cap_users.max())}명**, 평균 **{fmt(cap_users.mean())}명**만 남는다. 평균 **{pct(1-cap_users.mean()/full)} 감소**다. 제한이 사용자 이력과 평가 집단을 크게 바꾸는 현상을 이 데이터에서 직접 확인했다.
- 지정한 Linux 추천 후보 {fmt(coverage['total_linux_candidates'])}개 중 **{fmt(coverage['linux_candidates_covered'])}개({pct(coverage['linux_candidates_covered']/coverage['total_linux_candidates'])})**만 관측된다. 리뷰가 많아도 최신 게임·전체 후보를 대표하지 않는다.
- 기존 frank와 **{fmt(overlap['shared_users'])}계정·{fmt(overlap['shared_user_game_pairs'])}사용자–게임 쌍**이 겹친다. 이 EDA를 수행할 당시에는 Kieran 사용자별 학습·검증 파일이 없어 **학습·평가 계정 중복을 검사하지 못했다.** 후속 모델 평가에서는 계정을 분리하고 다시 학습했다.

## 1. 다운로드·파싱 검증

원본: `EDA/data/raw/forgemaster_v1/`. Kaggle 버전 1의 파일별 URL을 지정했다. 다운로드가 ZIP으로 반환되는 파일은 전체 멤버를 읽어 CRC를 확인한 뒤 CSV를 보존했다. 11개 파일의 열은 모두 다음 13개다.

`steamid`, `appid`, `voted_up`, `votes_up`, `votes_funny`, `weighted_vote_score`, `playtime_forever`, `playtime_at_review`, `num_games_owned`, `num_reviews`, `review`, `unix_timestamp_created`, `unix_timestamp_updated`.

`steamid`는 전 과정 문자열로 처리해 17자리 ID 정밀도를 보존했다. DuckDB의 strict 모드가 일부 파일에서 파서 오류를 냈으므로 열·구분자·따옴표를 명시하고 `strict_mode=false`, `ignore_errors=false`로 읽었다. **행을 버리거나 대체하지 않았다.** 별도로 Python CSV 파서로 11개 파일의 모든 레코드가 13열인지 확인했으며, strict 파싱 실패 파일은 **{csv_strict_failures}개**다. Python과 DuckDB의 전체·파일별 행 수가 일치한다. 따라서 DuckDB strict 실패 자체를 원본 CSV 손상으로 해석하지 않는다.

원본 행·계정·게임 수는 배포자가 적은 {fmt(15437471)} / {fmt(6976390)} / {fmt(8183)}과 {'모두 일치한다' if all(m['reconciliation'][key] for key in ['source_rows_match_publisher','source_users_match_publisher','source_apps_match_publisher']) else '일부 차이가 있다. 아래 JSON 참조'}.

## 2. 결측·중복·이상값

- 사용자–게임 반복 쌍: **{fmt(dup['repeated_pairs'])}개**, 중복으로 추가된 행 **{fmt(dup['extra_user_game_rows'])}개**.
- 긍정/부정 라벨이 충돌한 사용자–게임: **{fmt(dup['conflicting_pairs'])}쌍**, 해당 원본 행 **{fmt(dup['rows_in_conflicting_pairs'])}개**. 임의로 마지막 라벨을 선택하지 않고 충돌 쌍 전체를 제외했다. 동일 라벨 반복은 한 쌍으로 합쳤다.
- 정리 후 **{fmt(clean['rows'])}쌍·{fmt(clean['users'])}계정·{fmt(clean['apps'])}게임**이다. 이는 같은 사용자와 게임이 반복되는지 검사한 결과다. 모든 열이 완전히 같은 행을 센 값과는 다르다.
- 열별 결측·공백 및 숫자/불리언 변환 실패는 [열별 품질 검사표](tables/forgemaster/02_schema_quality.csv)에서 전체 행을 집계했다. 텍스트 결측·플레이타임 이상값 때문에 선호 라벨을 임의로 바꾸지 않았다.
- 리뷰 본문이 결측·공백인 행은 **{fmt(m['null_or_blank']['review'])}개({pct(m['null_or_blank']['review']/raw['rows'])})**다. ID·게임·추천 라벨 결측과 모든 숫자/불리언 변환 실패는 0개다.

{quality_table}

시간은 UTC로 해석했다. 작성 시각은 **{time_audit['first_created_utc']}–{time_audit['last_created_utc']}**다. **갱신 시각이 배포 시각 이후인 행 {fmt(time_audit['updated_after_publication_rows'])}개**, 분석일 이후인 행 {fmt(time_audit['updated_after_analysis_date_rows'])}개가 있으며 최대 갱신 시각은 **{time_audit['last_updated_utc']}**다. 이 미래 갱신 시각은 시간순 분할에 그대로 사용하면 안 된다. 평가 조건과 라벨 집계에는 갱신 시각을 사용하지 않았다. `playtime_*`는 Steam API 수치를 보존했고 Steam API의 분 단위로 분석했다. 배포 카드의 hours 표기를 그대로 가중치에 쓰면 단위 오류가 생길 수 있다. 현재 추천 점수와 평가 사용자 집계에는 플레이타임을 사용하지 않는다. `weighted_vote_score`는 리뷰 유용성 점수이며 게임의 선호 점수가 아니다. 선호 라벨은 `voted_up`이다.

## 3. 게임별 분포와 100개 상한

게임별 리뷰 수의 평균 **{m['reviews_per_game']['mean']:,.1f}**, 중앙값 **{fmt(gameq[1])}**, P90 **{fmt(gameq[3])}**, P99 **{fmt(gameq[5])}**, 최대 **{fmt(m['cap_shape']['maximum'])}**이다. 100개 이하 {fmt(m['cap_shape']['games_le_100'])}게임, 정확히 100개 {fmt(m['cap_shape']['games_exact_100'])}게임, 100개 초과 {fmt(m['cap_shape']['games_gt_100'])}게임이다. 전체 리뷰의 **{pct(m['full_top10_review_share'])}**가 상위 10개 게임에 몰려 있다.

리뷰마다 같은 가중치를 준 긍정 비율은 **{pct(clean['positive_rate'])}**, 게임별 긍정률의 단순 평균은 **{pct(m['cap_shape']['macro_positive_rate'])}**다. 게임을 같은 비중으로 보는지 리뷰를 같은 비중으로 보는지에 따라 수치가 달라진다. 전체 Steam 모집단의 평점으로 해석하지 않는다.

{top_table}

## 4. 사용자 이력·시간 범위

사용자별 관측 리뷰 수: 평균 **{m['reviews_per_user']['mean']:.2f}**, 중앙값 **{fmt(userq[1])}**, P90 **{fmt(userq[3])}**, P99 **{fmt(userq[5])}**, 최대 **{fmt(m['reviews_per_user']['maximum'])}**다. 한 게임만 관측된 사용자는 **{fmt(histories['singleton_users'])}명({pct(histories['singleton_users']/clean['users'])})**으로 기본 추천 평가에는 쓸 수 없다.

`num_reviews`보다 관측한 사용자–게임 수가 작은 계정은 **{fmt(histories['observed_less_than_reported_users'])}명**이다. 이 필드는 사용자 리뷰 수의 수집 당시의 값이며 서로 다른 게임을 수집한 시각도 다를 수 있다. 따라서 이 비율은 불완전 이력의 참고 지표이며, 사용자 전체 이력을 완전히 수집했다는 보장은 없다. 값이 반대로 큰 계정도 별도 기록했다.

리뷰 작성 연도는 **2010–2021년**이다. 2021년 이후 신규 게임·최근 사용자 행동 검증에는 사용할 수 없다. 2021년 자료를 더 뒤 시점의 Kieran으로 학습한 모델에 넣으면 시간상 미래 예측 검증이 되지 않는다. 같은 Steam 계정이 학습에 있는지도 별도로 확인해야 한다.

## 5. 현재 모델로 평가할 수 있는 범위

게임은 원 Steam App ID로만 연결했다. 공통 입력 게임 목록 {fmt(coverage['total_input_apps'])}개 중 **{fmt(coverage['input_apps_covered'])}개**, 지정한 Linux 추천 후보 {fmt(coverage['total_linux_candidates'])}개 중 **{fmt(coverage['linux_candidates_covered'])}개**가 이 데이터에 있다. Linux 판정은 기존 고정 참조를 따르며 현재 지원 상태를 새로 수집한 것은 아니다. Kieran 집계표와 겹치는 게임은 **{fmt(m['kieran_game_support_overlap']['apps'])}개**지만, 게임 중복은 학습·평가 사용자 중복 검사와 다르다.

입력은 고정 카탈로그 안의 긍정·부정 리뷰를 보존한다. 사용자당 관측된 긍정 Linux 게임 하나를 균등 무작위로 숨긴다고 가정하고, 남은 전체 입력 ≥5·긍정 입력 ≥1을 기본 조건으로 삼았다. 아래는 **EDA 단계에서 계산한 평가 가능 규모**다. 이 절의 수치는 추천 성능이 아니며 실제 정답 파일과 모델 평가는 후속 보고서에서 확인할 수 있다.

{elig_table}

기본 평가 조건을 만족한 사용자에게 숨기기 전 선택 가능한 정답은 **{fmt(m['full_eligible_history']['possible_target_pairs'])}쌍**이다. 이를 전부 동시에 숨기는 실험이 아니다. 정답 한 개를 숨긴 후 평균 입력 수는 **{m['full_eligible_history']['mean_inputs_after_holdout']:.2f}개**, 중앙값 **{fmt(m['full_eligible_history']['median_inputs_after_holdout'])}개**다.

## 6. 게임당 리뷰 100개 제한의 영향

정리한 사용자–게임 쌍에서 게임별 최대 100개를 균등 무작위로 남긴다. `hash(steamid, app_id, seed)` 순서와 문자열 ID 동률 처리로 재현하며, DuckDB 버전은 JSON에 기록했다. 게임 선택·연도·라벨·후보 조건을 고정하고 난수 시드 5개의 결과를 비교했다. **frank의 실제 API 정렬·리뷰어 선별을 복원한 실험은 아니다.**

{cap_table}

리뷰는 {fmt(clean['rows'])}쌍에서 매번 **{fmt(caps[0]['rows'])}쌍({pct(caps[0]['rows']/clean['rows'])} 유지)**으로 줄었다. 평가 가능한 사용자 수는 평균 {fmt(full)}→{fmt(cap_users.mean())}명으로 **{pct(cap_users.mean()/full)}만 유지**됐다. 여러 게임의 리뷰가 제한 표본에 함께 남은 사용자로 평가 대상이 바뀐다.

100개 초과 게임의 전체 관측 긍정률을 기준으로 제한 표본의 게임별 긍정률 절대오차를 계산했다. 표의 MAE는 **추천 Recall/NDCG나 모델 점수의 오차가 아니다.** 리뷰 가중 전체 긍정률과 인기 분포도 바뀐다. 따라서 frank와 다른 데이터의 평가 점수를 바로 비교해 모델 개선·악화를 판단하면 데이터 선택 효과와 모델 효과가 섞인다. 샘플링을 제거한다고 특정 모델의 점수가 반드시 상승하는 것은 아니다.

![데이터 분포 개요](../figures/forgemaster/01_overview.png)

![100개 제한의 분포 영향](../figures/forgemaster/02_cap_distribution.png)

## 7. frank와의 중복과 EDA 당시 검사 범위

frank는 기존 지정한 커밋 `75694cc07a6cdc69f48f9c316ca1e3e38a3f0f89`의 전체 {fmt(2631430)}행에서 식별 정보·라벨·시간 열 6개를 추출했다. 기존 추출 파일의 SHA256과 같은 **`{frank['projection_sha256']}`**이다. frank 본문이나 읽지 않은 원본 바이트까지 검증했다는 뜻은 아니다. 기존의 충돌 리뷰 제외 규칙을 적용한 frank 평가 가능 사용자는 **{fmt(frank['eligibility_after_existing_quarantine'][2]['eligible_users'])}명**으로 기존 보고된 24,342명과 대조한다.

{dimensions}

- 전체 계정 교집합 **{fmt(overlap['shared_users'])}명**; 사용자–게임 교집합 **{fmt(overlap['shared_user_game_pairs'])}쌍**; 같은 쌍에서 긍정/부정 불일치 **{fmt(overlap['shared_pair_label_disagreements'])}쌍**. 불일치는 시점별 평가 수정이나 수집 차이일 수 있어 임의로 합치지 않는다.
- 두 자료에서 모두 기본 평가 조건을 만족한 사용자는 **{fmt(elig_overlap['eligible_in_both'])}명**이다. Forgemaster 평가 가능 사용자 중 frank 전체 계정에도 있는 사람은 **{fmt(elig_overlap['source_eligible_users_in_any_frank'])}명**이다. 출처가 다르다는 사실은 독립성을 보장하지 않는다.
- forgemaster에는 `recommendationid`가 없다. 사용자–게임은 검사할 수 있지만 원 리뷰 ID 중복과 기존 296개 모호 리뷰 ID의 제외 규칙을 같은 방식으로 적용할 수 없다.
- **이 EDA 당시에는 Kieran 사용자·데이터 분할·모델 파일이 없었다.** 게임별 학습 사용자 수 CSV만으로는 계정 중복을 확인할 수 없었다. 따라서 이 분석만으로 Kieran과 사용자가 겹치지 않는다고 판단하지 않았다. 이후 원본 복원과 중복 계정 제외 재학습을 완료했다.

## 8. 권장 사용 방법

1. **보조 평가:** 기존 6,630개 Linux 후보와 입력·제외·정답 조건을 유지하고 forgemaster용 평가 계정·정답 목록과 검증 정보를 별도 파일에 저장해 고정한다. 관측되지 않은 후보도 원 후보 집합에서 없애지 않는다. 해당 데이터에서 정답으로 관측된 2,210개 게임 범위의 성능임을 밝힌다.
2. **독립성 확보:** Kieran 원 계정/검증·학습 분할을 복구해 정확한 SteamID로 대조한다. 새 테스트 계정이 기존 학습/검증에 있으면 계정 전체를 제외하고 필요한 학습·모델 선택을 다시 수행한다. 이미 쓴 frank와의 중복 계정은 별도 집단으로 보고한다.
3. **100개 제한 영향 비교:** 학습한 모델을 그대로 사용해 같은 데이터의 리뷰 수 제한 전후 비교, 같은 사용자 집단의 입력 축소 비교, 공통 게임·연도·긍정 학습 사용자 수 구간별 결과를 나눠 본다. 실제 frank의 수집 방식 차이는 별도 한계로 둔다. EDA에서 관측한 적격 사용자 감소만으로 Recall 하락 폭을 추정하지 않는다.
4. **지표 보고:** Recall@10/NDCG@10과 함께 평가 사용자 수, 입력 길이, 정답 인기 구간, 게임별 평균 지표, 추천 없음 비율을 보고한다. 이 EDA 보고서는 추천 성능을 계산하기 전의 분석이다. 모델 지표는 후속 보고서에 있다.

## 결과 파일과 재현 방법

- [전체 수치·출처·검증 JSON](forgemaster_eda_metrics.json)
- [실행 결과가 포함된 노트북](../notebooks/02_forgemaster_eda.ipynb)
- 집계표: `reports/tables/forgemaster/`, 그래프: `figures/forgemaster/`
- 원본 CSV·다운로드·검증 기록: `data/raw/forgemaster_v1/`
- 정리한 사용자–게임 쌍·평가 가능 계정·작업 DB: `data/processed/forgemaster_v1/` (계정 ID 포함 파일은 로컬 Git 제외 경로에만 저장)
- frank 비교용 추출 파일·검증 정보: `data/processed/forgemaster_eda/`

저장소 루트 PowerShell에서 실행한다. 기존 분할과 모델을 갱신하지 않는다.

```powershell
.venv/Scripts/python -m pip install duckdb==1.4.3 pandas numpy matplotlib pyarrow nbformat nbclient ipykernel
.venv/Scripts/python EDA/scripts/download_forgemaster.py
.venv/Scripts/python EDA/scripts/fetch_frank_review_comparison.py
.venv/Scripts/python EDA/scripts/verify_forgemaster_csv.py
.venv/Scripts/python EDA/scripts/eda_forgemaster.py
.venv/Scripts/python EDA/scripts/audit_forgemaster_results.py
.venv/Scripts/python EDA/scripts/build_forgemaster_report.py
```

전체 행 수, 파서별 파일 행 수, 게임·사용자·연도별 합계, 사용자–게임 쌍의 중복 여부와 5개 제한 표본의 행 수가 같은지 확인했다. 상세 통과 여부와 모든 원본 파일 SHA256은 JSON에 보존했다.
"""
    REPORT.write_text(report,encoding="utf-8")
    nb=nbformat.v4.new_notebook()
    nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                 "language_info":{"name":"python","version":"3.13"}}
    nb.cells=[nbformat.v4.new_markdown_cell("# Forgemaster Steam Reviews v1 데이터 분석\n\n2026-11-08. 원본 11개 파일 전체를 분석한 결과. 실제 학습/정답 분할 및 추천 지표 평가는 수행하지 않았습니다.\n\n전체 설명: [보고서](../reports/forgemaster_eda.md). 재수집·전수 집계 명령은 보고서 마지막을 참조하세요."+followup_notebook),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nEDA = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd() / 'EDA'\nm = json.loads((EDA / 'reports/forgemaster_eda_metrics.json').read_text('utf-8'))\nT = EDA / 'reports/tables/forgemaster'\ndisplay(pd.DataFrame([m['raw'],m['clean']],index=['raw','clean']))"),
        nbformat.v4.new_markdown_cell("## 열별 품질 검사\n\nSteamID는 문자열로 보존합니다. 라벨 충돌 사용자–게임은 전체 제외하고 동일 라벨 반복은 한 쌍으로 합쳤습니다. 두 파서의 파일별 전체 행 수를 대조했습니다."),
        nbformat.v4.new_code_cell("display(pd.read_csv(T / '02_schema_quality.csv'))\ndisplay(pd.Series(m['duplicates'],name='사용자–게임 중복 검사'))\ndisplay(pd.Series(m['quality'],name='품질 검사 결과'))\ndisplay(pd.Series(m['additional_saved_output_audit']['timestamps'],name='날짜 검사 결과'))\ndisplay(pd.Series(m['reconciliation'],name='집계 일치 검사'))"),
        nbformat.v4.new_markdown_cell("## 게임·사용자·시간 분포"),
        nbformat.v4.new_code_cell("display(pd.DataFrame([m['reviews_per_game'],m['reviews_per_user']],index=['per game','per user']))\ndisplay(pd.read_csv(T / '03_game_support.csv').head(20))\ndisplay(pd.read_csv(T / '06_review_years.csv'))\ndisplay(Image(filename=str(EDA / 'figures/forgemaster/01_overview.png')))"),
        nbformat.v4.new_markdown_cell("## 입력·추천 게임 목록과 평가 가능 사용자\n\n정답 한 개를 숨긴 뒤 최소 전체 입력 5개·긍정 입력 1개가 기본 조건입니다. 아래는 EDA 단계에서 계산한 평가 가능 사용자 수입니다."),
        nbformat.v4.new_code_cell("display(pd.Series(m['catalog_coverage'],name='게임 목록의 데이터 포함 범위'))\ndisplay(pd.read_csv(T / '07_evaluation_eligibility.csv'))\ndisplay(pd.Series(m['full_eligible_history'],name='평가 대상자의 리뷰 이력'))"),
        nbformat.v4.new_markdown_cell("## 게임당 최대 100개 제한: 난수 시드 5개\n\n균등 무작위 제한은 frank의 실제 API 정렬·선별을 복원하지 않습니다. 게임별 긍정률 MAE는 추천 성능 지표가 아닙니다."),
        nbformat.v4.new_code_cell("display(pd.read_csv(T / '09_cap_sensitivity.csv'))\ndisplay(Image(filename=str(EDA / 'figures/forgemaster/02_cap_distribution.png')))"),
        nbformat.v4.new_markdown_cell("## frank 중복과 학습·평가 계정 중복 검사\n\n정확한 SteamID와 App ID를 대조했습니다. 이 EDA 당시에는 Kieran 계정별 분할 파일이 없어 학습·평가 계정 중복을 확인하지 못했습니다. 후속 모델 평가에서는 계정을 분리했습니다."),
        nbformat.v4.new_code_cell("display(pd.Series(m['frank_comparison']['overlap'],name='계정·게임 ID 중복'))\ndisplay(pd.Series(m['frank_comparison']['source_eligible_overlap'],name='평가 가능 사용자 중복'))\ndisplay(pd.Series(m['training_identity_audit']))"),
        nbformat.v4.new_markdown_cell("## 결론\n\n100개 제한의 분포·이력 영향을 살피는 **리뷰 수 제한의 영향을 확인하는 추가 평가 데이터**입니다. 모델 평가에는 Kieran 학습 계정과 중복되지 않게 분리한 데이터를 사용합니다. 2021년 자료·일부 게임만 포함하므로 최신 게임 및 전체 후보의 성능을 대표하지 않습니다.")]
    notebook=EDA / "notebooks/02_forgemaster_eda.ipynb"
    NotebookClient(nb,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(EDA / 'notebooks')}}).execute()
    nbformat.write(nb,notebook)
    # Paired text view is also executable as a result-inspection script.
    parts=[]
    for cell in nb.cells:
        if cell.cell_type=="markdown":
            parts.append("# %% [markdown]\n"+"\n".join("# "+line if line else "#" for line in cell.source.splitlines()))
        else:
            parts.append("# %%\n"+cell.source)
    notebook.with_suffix(".py").write_text("\n\n".join(parts)+"\n",encoding="utf-8")
    print(f"Wrote {REPORT}\nExecuted {notebook}")


if __name__=="__main__":
    main()
