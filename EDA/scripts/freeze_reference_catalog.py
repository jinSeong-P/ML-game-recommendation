"""Freeze non-personal catalog/coverage and previously audited review mappings.

Only needed to rebuild the committed reference snapshots from archived inputs.
No former A/B/C user interactions become model-training data.
"""
import json
from pathlib import Path

import duckdb

from eda_large_review_frank import digest, quote

EDA = Path(__file__).resolve().parents[1]


def main():
    old = EDA / "legacy/data/processed"
    if not old.exists():  # Initial migration, before moving the archives.
        old = EDA / "data/processed"
    ref = EDA / "data/reference"
    ref.mkdir(parents=True, exist_ok=True)
    kieran = EDA / "data/processed/kieran_original/eda_work.duckdb"
    if not kieran.exists():
        kieran = old / "candidate_identity/kieran_original/eda_work.duckdb"
    frank = EDA / "data/processed/frank/reviews_identity_columns.parquet"
    if not frank.exists():
        frank = old / "candidate_identity/frankjc2022/reviews_identity_columns.parquet"
    metric = json.loads((EDA/'reports/kieran_original_frank_eda_metrics.json').read_text('utf-8'))
    baseline = json.loads((EDA/'legacy/reports/kieran_original_frank_eda_metrics.json').read_text('utf-8')) if (EDA/'legacy/reports/kieran_original_frank_eda_metrics.json').exists() else metric
    for item in baseline['reference_catalog_inputs']:
        if digest(old/item['file']) != item['sha256']:
            raise ValueError(f'Archived catalog input changed: {item["file"]}')
    frank_manifest = json.loads((frank.parent/'projection_manifest.json').read_text('utf-8'))
    if frank_manifest['commit'] != baseline['frank_provenance']['commit']:
        raise ValueError('frank commit differs from the original catalog audit')
    expected_frank = baseline['frank_provenance']['projection_sha256']
    if digest(frank) != expected_frank or frank_manifest['files']['reviews']['local_projection']['sha256'] != expected_frank:
        raise ValueError('frank projection differs from the original catalog audit')
    old_reference = json.loads((ref/'manifest.json').read_text('utf-8')) if (ref/'manifest.json').exists() else None
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"""CREATE TABLE catalog AS SELECT g.app_id,g.linux,
        r.app_id IS NOT NULL reviewed FROM read_parquet({quote(old/'A_games.parquet')}) g
        LEFT JOIN (SELECT DISTINCT app_id FROM read_parquet({quote(old/'A_recommendations.parquet')})) r USING(app_id)""")
    con.execute(f"""CREATE TABLE content_catalog AS WITH a AS (
        SELECT app_id,length(trim(coalesce(description,'')))>0 has_description,
            coalesce(array_length(tags),0)>0 has_tags
        FROM read_parquet({quote(old/'A_games_metadata.parquet')})),
        c AS (SELECT appid app_id,
            length(trim(coalesce(nullif(about_the_game,''),nullif(detailed_description,''),short_description,'')))>0 has_description,
            coalesce(trim(tags),'') NOT IN ('','[]','{{}}','None','null') has_tags
            FROM read_parquet({quote(old/'C_games_2025.parquet')}))
        SELECT coalesce(a.app_id,c.app_id) app_id,a.app_id IS NOT NULL in_A_metadata,
            c.app_id IS NOT NULL in_C_metadata,
            coalesce(a.has_description,false) OR coalesce(c.has_description,false) has_description,
            coalesce(a.has_tags,false) OR coalesce(c.has_tags,false) has_tags
        FROM a FULL OUTER JOIN c USING(app_id)""")
    con.execute(f"CREATE TABLE frank_raw AS SELECT cast(recommendationid AS UBIGINT) review_key,appid app_id FROM read_parquet({quote(frank)})")
    for alias, path in [('a',old/'candidate_identity/artermiloff_2024/eda_work.duckdb'),('b',kieran)]:
        con.execute(f"ATTACH {quote(path)} AS src_{alias} (READ_ONLY)")
    con.execute("""CREATE TABLE review_app_quarantine AS
        SELECT f.review_key FROM frank_raw f JOIN src_a.main.review_keys k USING(review_key)
            WHERE k.min_app<>k.max_app OR k.min_user<>k.max_user
        UNION SELECT f.review_key FROM frank_raw f JOIN src_b.main.review_keys k USING(review_key)
            WHERE k.min_app<>k.max_app OR k.min_user<>k.max_user
        UNION SELECT f.review_key FROM src_a.main.valid_raw r JOIN frank_raw f USING(review_key) WHERE r.app_id<>f.app_id
        UNION SELECT f.review_key FROM src_b.main.valid_raw r JOIN frank_raw f USING(review_key) WHERE r.app_id<>f.app_id""")
    assert con.execute('SELECT count(*) FROM review_app_quarantine').fetchone()[0] == 296
    for table in ['catalog','content_catalog','review_app_quarantine']:
        order = 'review_key' if table == 'review_app_quarantine' else 'app_id'
        staged = ref/(table+'.staging.parquet')
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY {order}) TO {quote(staged)} (FORMAT PARQUET)")
        if old_reference and digest(staged) != old_reference['files'][table+'.parquet']['sha256']:
            staged.unlink()
            raise ValueError(f'Rebuilt snapshot differs from frozen baseline: {table}')
        staged.replace(ref/(table+'.parquet'))
    previous = EDA/'reports/large_original_frank_comparison_metrics.json'
    if not previous.exists():
        previous = EDA/'legacy/reports/large_original_frank_comparison_metrics.json'
    manifest = {
        'version':'kieran-frank-reference-v1',
        'purpose':'Frozen EDA eligibility, Linux flags, metadata availability and known ambiguous review mappings; no account IDs or training interactions',
        'former_catalog_inputs':baseline['reference_catalog_inputs'],
        'historical_mapping_audit_sha256':digest(previous),
        'frank_commit':metric['frank_provenance']['commit'],
        'quarantine_scope':'Known review-to-App-ID ambiguity union from the completed two-source audit. Rounded artermiloff owner-ID mismatches alone are NOT used.',
        'content_scope':'Availability flags only, not game text, tags or model-ready TF-IDF features',
        'files':{table+'.parquet':{'sha256':digest(ref/(table+'.parquet')),
            'rows':con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}
            for table in ['catalog','content_catalog','review_app_quarantine']}}
    (ref/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    con.close()
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
