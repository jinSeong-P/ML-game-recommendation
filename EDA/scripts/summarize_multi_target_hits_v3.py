"""Add descriptive hit-count metrics to saved v3 two-target evaluations."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from eda_large_review_frank import EDA, digest, quote

COHORTS = EDA / 'data/processed/modeling/backlog_v3/cohorts_v3_final'
EVALUATIONS = EDA / 'data/processed/modeling/improvement_v3/evaluations'
LABELS = {
    'item_knn_control': '기본 Item-KNN',
    'activity_inverse_degree_cosine_k100': '활동량 보정 Item-KNN',
    'content_control': '소개글 bigram + 태그·장르',
    'metadata_extended_tfidf': '확장 메타데이터 TF-IDF',
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_files(folder, manifest):
    for name, info in manifest['files'].items():
        if digest(folder / name) != info['sha256']:
            raise ValueError(f'Saved artifact changed: {folder / name}')


def summarize():
    started = time.perf_counter()
    rows, sources = [], []
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=2')
    try:
        for cohort in ('multi2_validation', 'multi2_frank'):
            source = COHORTS / cohort
            cm = load(source / 'manifest.json')
            if cm['status'] != 'complete' or cm['parameters']['target_count'] != 2:
                raise ValueError('Only completed two-target cohorts are supported')
            verify_files(source, cm)
            users = pq.read_table(source / 'users.parquet').to_pydict()
            expected_users = np.asarray(users['user_key'], dtype=np.uint64)
            if (len(expected_users) != cm['users'] or len(np.unique(expected_users)) != len(expected_users)
                    or np.any(np.asarray(users['n_total_inputs']) < 5)
                    or np.any(np.asarray(users['n_positive_inputs']) < 1)):
                raise ValueError('Saved cohort eligibility changed')
            con.execute(f'CREATE OR REPLACE VIEW targets AS SELECT * FROM read_parquet({quote(source / "targets.parquet")})')
            if con.execute('SELECT count(*) FROM (SELECT user_key FROM targets GROUP BY user_key HAVING count(*)<>2 OR count(DISTINCT app_id)<>2)').fetchone()[0]:
                raise ValueError('Expected exactly two distinct targets per user')
            for variant, label in LABELS.items():
                folder = EVALUATIONS / cohort / variant
                manifest = load(folder / 'manifest.json')
                if manifest['status'] != 'complete' or any(manifest['violations'].values()):
                    raise ValueError(f'Incomplete or invalid saved evaluation: {folder}')
                verify_files(folder, manifest)
                if manifest['provenance']['cohort_hashes'] != {k: v['sha256'] for k, v in cm['files'].items()}:
                    raise ValueError('Evaluation and cohort provenance differ')
                data = pq.read_table(folder / 'per_user.parquet').to_pydict()
                hits = np.asarray(data['retrieved_targets'], dtype=np.int64)
                lengths = np.asarray(data['recommendation_count'], dtype=np.int64)
                if (not np.array_equal(data['user_key'], expected_users)
                        or np.any(np.asarray(data['hidden_targets']) != 2)
                        or np.any((hits < 0) | (hits > 2) | (lengths < hits) | (lengths > 10))):
                    raise ValueError('Invalid saved per-user hit counts')
                con.execute(f'CREATE OR REPLACE VIEW outcomes AS SELECT * FROM read_parquet({quote(folder / "per_user.parquet")})')
                con.execute(f'CREATE OR REPLACE VIEW recs AS SELECT * FROM read_parquet({quote(folder / "recommendations.parquet")})')
                checks = {
                    'duplicate_recommendations': 'SELECT count(*) FROM (SELECT user_key,app_id FROM recs GROUP BY ALL HAVING count(*)<>1)',
                    'invalid_recommendations': 'SELECT count(*) FROM recs WHERE rank<1 OR rank>10 OR score<=0 OR NOT isfinite(score)',
                    'unknown_recommendation_users': 'SELECT count(*) FROM recs ANTI JOIN outcomes USING(user_key)',
                    'hit_replay_mismatches': '''SELECT count(*) FROM outcomes o LEFT JOIN
                        (SELECT r.user_key,count(*) hits,sum(1/log2(r.rank+1)) dcg FROM recs r JOIN targets t USING(user_key,app_id) GROUP BY r.user_key) h USING(user_key)
                        WHERE o.retrieved_targets<>coalesce(h.hits,0)
                           OR abs(o.ndcg_at_10-coalesce(h.dcg,0)/(1+1/log2(3)))>1e-12''',
                    'length_replay_mismatches': '''SELECT count(*) FROM outcomes o LEFT JOIN
                        (SELECT user_key,count(*) n FROM recs GROUP BY user_key) r USING(user_key)
                        WHERE o.recommendation_count<>coalesce(r.n,0)''',
                }
                audit = {name: int(con.execute(sql).fetchone()[0]) for name, sql in checks.items()}
                if any(audit.values()):
                    raise ValueError(f'Saved recommendation replay failed: {audit}')
                row = dict(cohort=cohort, variant=variant, users=len(hits),
                    users_zero_hits=int(np.sum(hits == 0)), users_one_hit=int(np.sum(hits == 1)),
                    users_two_hits=int(np.sum(hits == 2)), hit_rate_ge1_at10=float(np.mean(hits >= 1)),
                    hit_rate_ge2_at10=float(np.mean(hits >= 2)), mean_hits_at10=float(hits.mean()),
                    precision_at10=float(np.mean(hits / 10)), recall_at10=float(np.mean(hits / 2)),
                    ndcg_at10=float(np.mean(data['ndcg_at_10'])), empty_users=int(np.sum(lengths == 0)))
                for key, saved in [('recall_at10', 'Recall@10'), ('ndcg_at10', 'NDCG@10'), ('hit_rate_ge1_at10', 'HitRate@10')]:
                    if abs(row[key] - manifest['metrics'][saved]) > 1e-12:
                        raise ValueError(f'Existing metric mismatch: {key}')
                if (not np.allclose(data['recall_at_10'], hits / 2, atol=1e-12, rtol=0)
                        or abs(row['mean_hits_at10'] - row['hit_rate_ge1_at10'] - row['hit_rate_ge2_at10']) > 1e-12
                        or row['empty_users'] != manifest['metrics']['empty_users']):
                    raise ValueError('Hit-count identities or existing empty-user count failed')
                rows.append(row)
                sources.append(dict(cohort=cohort, variant=variant,
                    cohort_manifest_sha256=digest(source / 'manifest.json'),
                    evaluation_manifest_sha256=digest(folder / 'manifest.json'),
                    files=manifest['files'], audit=audit))
    finally:
        con.close()
    elapsed = time.perf_counter() - started
    output = EDA / 'reports/multi_target_hit_rates_v3'
    result = dict(status='complete_saved_two_target_reporting_only', code_sha256=digest(Path(__file__)),
        hidden_targets_per_user=2, elapsed_seconds=elapsed, retrained=False, rescored=False,
        model_selection_changed=False, outcomes=rows, sources=sources)
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (EDA / 'reports/tables/multi_target_hit_rates_v3.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 정답 2개 평가의 적중 수 추가 지표', '',
        '저장된 v3 사용자별 결과와 추천 목록으로 추가 집계했다. 재학습·추천 재생성·모델 선택 변경은 하지 않았다. 현재 v4 확인 평가와 사용자·모델·정답이 다르므로 개선량을 직접 비교하지 않는다.', '',
        '각 사용자의 긍정 Linux 정답 2개를 동시에 숨기고, 남은 전체 입력 ≥5·긍정 입력 ≥1인 기존 집단을 그대로 쓴다. 후보 6,630개, 관측 긍정·부정 게임 제외, 양수 개인화 점수만 최대 10개, 빈자리 보완 없음, 빈 목록도 분모 유지 규칙을 따른다.', '',
        '- **1개 이상 적중률:** 숨긴 두 정답 중 하나 이상을 Top 10에서 찾은 사용자 비율.',
        '- **2개 모두 적중률:** 숨긴 두 정답을 모두 Top 10에서 찾은 사용자 비율.',
        '- **평균 적중 수:** 사용자당 찾은 정답 수(0–2)의 평균.',
        '- **Precision@10:** 적중 수 ÷ 10의 사용자 평균. 목록이 짧아도 분모는 10이다.',
        '- **Recall@10:** 적중 수 ÷ 2의 사용자 평균. 복수 정답에서는 1개 이상 적중률과 다르다.', '',
        'Precision@10은 숨긴 정답이 2개라 최대 20%다. 관측되지 않은 다른 추천 게임을 좋아할 가능성은 측정하지 않으므로 실제 추천 만족도나 모든 관련 게임에 대한 정밀도로 해석하지 않는다. 리뷰가 충분한 사용자 집단에 대한 보조 평가이며, 기존 단일 정답 평가와 절대값을 직접 비교하지 않는다. 기존 frank도 이미 확인한 참조 자료이며 새 독립 최종 테스트가 아니다.', '']
    for cohort, title in [('multi2_validation', '내부 검증 — 217,720명'), ('multi2_frank', 'frank 참조 평가 — 14,819명')]:
        lines += [f'## {title}', '', '| 모델 | 1개 이상 | 2개 모두 | 두 개 모두 찾은 사용자 | 평균 적중 수 | Precision@10 | Recall@10 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['cohort'] != cohort:
                continue
            lines.append(f"| {LABELS[row['variant']]} | {row['hit_rate_ge1_at10']:.2%} | {row['hit_rate_ge2_at10']:.2%} | {row['users_two_hits']:,} | {row['mean_hits_at10']:.4f} | {row['precision_at10']:.2%} | {row['recall_at10']:.2%} |")
        lines.append('')
    lines += ['## 검증', '',
        '8개 평가의 원본 파일 해시와 집단 연결을 확인했다. 저장 추천 목록을 정답과 다시 조인해 사용자별 적중 수·NDCG·목록 길이를 재현했으며, 중복 추천·잘못된 점수/순위·미등록 사용자 검사 위반은 모두 0건이다. 기존 Recall·NDCG·1개 이상 적중률·빈 목록 수와 일치했다. 평균 적중 수 = 1개 이상 적중률 + 2개 모두 적중률도 확인했다.', '',
        f'파일 검사·추천 목록 검산·집계 소요 시간은 {elapsed:.2f}초다. 모델 학습이나 추론 시간은 포함하지 않는다.', '',
        '[전체 수치 CSV](tables/multi_target_hit_rates_v3.csv) · [출처 해시와 검산 JSON](multi_target_hit_rates_v3.json) · [재현 스크립트](../scripts/summarize_multi_target_hits_v3.py) · [백로그](../docs/modeling_backlog.md)', '',
        '재현: 저장소 루트에서 `EDA/.venv/Scripts/python.exe EDA/scripts/summarize_multi_target_hits_v3.py`를 실행한다. 원본 v3 평가 파일은 읽기 전용으로 사용한다.', '']
    output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], evaluations=len(rows), elapsed_seconds=elapsed, outcomes=rows), ensure_ascii=False))


if __name__ == '__main__':
    summarize()
