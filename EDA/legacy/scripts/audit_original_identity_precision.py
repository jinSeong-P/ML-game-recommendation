"""Diagnose original artermiloff ID rounding; never use floats to repair IDs.

The float round-trip here is a diagnostic comparison against exact trusted IDs.
Outputs are aggregate-only. Original CSV tokens are checked against projections.
"""

import csv
import io
import json
import zipfile
from pathlib import PurePosixPath

import duckdb

from eda_large_review_frank import EDA, many, one, quote


def main():
    folder=EDA/'data/processed/candidate_identity/artermiloff_2024'
    con=duckdb.connect(str(folder/'eda_work.duckdb'),read_only=True)
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=3')
    output={'purpose':'Source identity diagnosis; float conversion is only a test of rounding, never an ID recovery method',
            'artermiloff_identity_usable_for_exact_account_partition':False}
    output['whole_source_residues']=one(con,"""SELECT count(*) identity_valid_rows,
        count(*) FILTER(WHERE user_key%16=0) author_ids_divisible_by_16,
        count(*) FILTER(WHERE user_key%64=0) author_ids_divisible_by_64 FROM valid_raw""")
    output['matched_original_review_rows']=one(con,"""SELECT count(*) matched_rows,
        count(*) FILTER(WHERE r.user_key<>f.user_key) different_author_ids,
        count(*) FILTER(WHERE r.user_key=cast(cast(f.user_key AS DOUBLE) AS UBIGINT)) matches_float64_roundtrip_of_exact_frank_id,
        count(*) FILTER(WHERE abs(cast(r.user_key AS HUGEINT)-cast(f.user_key AS HUGEINT))<=8) author_difference_at_most_8
        FROM valid_raw r JOIN frank_raw f USING(review_key)""")
    output['author_delta_histogram']=many(con,"""SELECT
        cast(r.user_key AS HUGEINT)-cast(f.user_key AS HUGEINT) author_id_delta,count(*) n_rows
        FROM valid_raw r JOIN frank_raw f USING(review_key) GROUP BY 1 ORDER BY 1""")
    output['distinct_matched_review_ids']=one(con,"""SELECT count(DISTINCT r.review_key) matched_original_review_ids,
        count(DISTINCT r.review_key) FILTER(WHERE r.user_key<>f.user_key) different_author_original_review_ids
        FROM valid_raw r JOIN frank_raw f USING(review_key)""")
    output['known_account_collisions_in_frank_matched_subset']=one(con,"""WITH buckets AS (
        SELECT r.user_key,count(DISTINCT f.user_key) real_accounts FROM valid_raw r
        JOIN frank_raw f USING(review_key) GROUP BY r.user_key)
        SELECT count(*) FILTER(WHERE real_accounts>1) rounded_ids_representing_multiple_real_accounts,
        coalesce(sum(real_accounts) FILTER(WHERE real_accounts>1),0) real_accounts_in_colliding_buckets,
        max(real_accounts) maximum_known_real_accounts_per_rounded_id FROM buckets""")
    # Check the actual ZIP text, independently of our ingestion conversion.
    archive_path=EDA/'data/raw/candidate_review/artermiloff_2024/steam-games-reviews-2024-v1.zip'
    checked_files=checked_rows=projection_mismatches=frank_mismatches=0
    with zipfile.ZipFile(archive_path) as archive:
        infos={int(PurePosixPath(i.filename).stem):i for i in archive.infolist()
               if PurePosixPath(i.filename).stem.isdigit() and i.file_size<100_000}
        con.register('small_apps',__import__('pyarrow').table({'app_id':list(infos)}))
        examples=con.execute("""SELECT DISTINCT r.app_id FROM valid_raw r
            JOIN small_apps USING(app_id) JOIN frank_raw f USING(review_key)
            WHERE r.user_key<>f.user_key ORDER BY r.app_id LIMIT 3""").fetchall()
        for (app,) in examples:
            reference={str(r): (str(u),str(f)) for r,u,f in con.execute(f"""SELECT r.review_key,r.user_key,f.user_key
                FROM valid_raw r JOIN frank_raw f USING(review_key) WHERE r.app_id={int(app)}""").fetchall()}
            with archive.open(infos[app]) as stream:
                reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))
                for row in reader:
                    match=reference.get(row['recommendationid'])
                    if match is None:continue
                    checked_rows+=1
                    projection_mismatches+=row['author_steamid']!=match[0]
                    frank_mismatches+=row['author_steamid']!=match[1]
            checked_files+=1
    output['direct_original_zip_token_check']={'csv_files':checked_files,'matched_review_rows':checked_rows,
        'original_csv_author_token_different_from_local_projection':projection_mismatches,
        'original_csv_author_token_different_from_exact_frank':frank_mismatches}
    # A trusted review-ID -> exact owner mapping can expose held-out accounts
    # that the rounded author anti-join misses, beyond common frank review IDs.
    trusted=EDA/'data/processed/candidate_identity/kieran_original/eda_work.duckdb'
    con.execute(f'ATTACH {quote(trusted)} AS trusted (READ_ONLY)')
    output['known_test_accounts_left_in_naive_artermiloff_preview']=one(con,"""WITH trusted_test AS (
        SELECT k.review_key,k.min_user exact_user FROM trusted.main.review_keys k
        SEMI JOIN test_users t ON k.min_user=t.user_key WHERE k.min_user=k.max_user)
        SELECT count(DISTINCT r.review_key) known_test_account_review_ids_remaining_after_nominal_test_exclusion,
        count(DISTINCT k.exact_user) known_real_test_accounts_remaining,
        count(DISTINCT r.review_key) FILTER(WHERE NOT u.is_validation) known_test_account_review_ids_remaining_after_validation
        FROM compatible_raw r SEMI JOIN clean s USING(user_key,app_id)
        JOIN trusted_test k USING(review_key) JOIN user_stats u ON r.user_key=u.user_key
        WHERE NOT u.is_test""")
    output['decision']='REJECT_ORIGINAL_ARTERMILOFF_FOR_EXACT_USER_SPLIT; original integer-looking CSV IDs carry float64 rounding. Nominal zero intersections are insufficient. Kieran owner mapping may recover only rows with a trusted original review-ID match; converting rounded IDs to strings cannot recover them.'
    path=EDA/'reports/original_identity_precision_metrics.json'
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    con.close()


if __name__=='__main__':
    main()
