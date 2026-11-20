"""Check content source equivalence and independently audit sampled train membership."""
import json
import duckdb
import pyarrow.parquet as pq
from eda_large_review_frank import EDA,digest,quote
from experiment_v3_common import ROOT,V2,SPLIT,write_json


def run():
    pairs=[]
    for name in ['T2_unigram_bigram','metadata_channel']:
        a,b=V2/'tfidf'/name,V2/'tfidf_rebuild_final2'/name
        # Historical manifests can differ in provenance; the actual features,
        # vocabulary, IDF, and row mapping used for inference must be identical.
        m=json.loads((a/'manifest.json').read_text('utf-8'))
        for rel in m['files']:
            if not (b/rel).is_file():raise FileNotFoundError(b/rel)
            left,right=digest(a/rel),digest(b/rel)
            if left!=right:raise ValueError('v2 content rebuild differs: '+name+'/'+rel)
            pairs.append({'channel':name,'file':rel,'sha256':left})
    folder=ROOT/'content/train_review_body_tfidf'
    m=json.loads((folder/'manifest.json').read_text('utf-8'))
    for rel,info in m['files'].items():
        if digest(folder/rel)!=info['sha256']:raise ValueError('Review body artifact changed: '+rel)
    con=duckdb.connect()
    try:
        con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
        con.execute(f'CREATE TABLE sample AS SELECT * FROM read_parquet({quote(folder/"review_sample_lineage.parquet")})')
        n,unique_reviews,max_count,bad_rank=con.execute('''SELECT (SELECT count(*) FROM sample),
            (SELECT count(DISTINCT review_key) FROM sample),
            (SELECT max(n) FROM(SELECT count(*) n FROM sample GROUP BY app_id)),
            (SELECT count(*) FROM sample WHERE sample_rank<1 OR sample_rank>100)''').fetchone()
        # The bounded sample is the join build side. Scan accepted train lineage
        # rather than constructing a hash table of all 98.9 million train rows.
        matched=con.execute(f'''SELECT count(DISTINCT s.review_key) FROM
            read_parquet({quote(SPLIT/'train/review_lineage.parquet')}) t
            JOIN sample s ON t.review_key=s.review_key AND t.app_id=s.app_id''').fetchone()[0]
        heldout={}
        for cohort in ['validation','test']:
            p=SPLIT/cohort/'review_lineage.parquet'
            if p.exists():
                heldout[cohort]=con.execute(f'''SELECT count(*) FROM read_parquet({quote(p)}) t
                    JOIN sample s ON t.review_key=s.review_key''').fetchone()[0]
    finally:con.close()
    if n!=unique_reviews or n!=matched or max_count>100 or bad_rank or any(heldout.values()):raise ValueError('Invalid review sample membership')
    if n!=m['summary']['sampled_review_count']:raise ValueError('Review count differs from manifest')
    result={'status':'passed','code_sha256':digest(__file__),
        'historical_feature_byte_equivalence':pairs,
        'review_body':{'manifest_sha256':digest(folder/'manifest.json'),'sample_rows':n,
            'distinct_review_ids':unique_reviews,'matched_accepted_train_review_and_app':matched,
            'max_reviews_per_app':max_count,'bad_sample_rank':bad_rank,'heldout_review_matches':heldout,
            'tuple_join_mismatches_recorded_by_builder':m['summary']['sample_lineage_mismatches'],
            'scope':'Independent review/app membership audit plus builder exact user/review/app tuple audit; no heldout review text used for fit.'}}
    write_json(ROOT/'content_source_audit.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
