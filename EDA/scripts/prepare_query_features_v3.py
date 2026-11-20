"""Fetch pinned extra review columns; fit query-only language/year features."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from eda_large_review_frank import EDA,digest,quote,log
from fetch_frank_identity_columns import RangeFile
from experiment_v3_common import ROOT,SPLIT,catalog,write_json


def fetch_frank_features():
    out=ROOT/'query_context/frank_review_features.parquet'
    if out.exists():
        m=json.loads(out.with_suffix('.json').read_text('utf-8'))
        if digest(out)!=m['sha256']:raise ValueError('Frank features changed')
        return out
    out.parent.mkdir(parents=True,exist_ok=True)
    source=json.loads((EDA/'data/processed/frank/projection_manifest.json').read_text('utf-8'))
    info=source['files']['reviews']
    remote=RangeFile(info['source_url'],info['source_bytes'])
    parquet=pq.ParquetFile(remote)
    columns=['appid','recommendationid','author_steamid','voted_up','language',
             'author_playtime_forever','author_playtime_at_review']
    rows=0
    with pq.ParquetWriter(out,parquet.schema_arrow.empty_table().select(columns).schema,compression='zstd') as writer:
        for i in range(parquet.num_row_groups):
            table=parquet.read_row_group(i,columns=columns)
            writer.write_table(table);rows+=table.num_rows
            log(f'Pinned frank feature projection: group {i+1}/{parquet.num_row_groups}')
    con=duckdb.connect()
    try:
        con.execute('SET threads=2')
        original=EDA/'data/processed/frank/reviews_identity_columns.parquet'
        bad=con.execute(f'''SELECT count(*) FROM (
            (SELECT appid,recommendationid,author_steamid,voted_up FROM read_parquet({quote(out)})
             EXCEPT ALL SELECT appid,recommendationid,author_steamid,voted_up FROM read_parquet({quote(original)}))
             UNION ALL
            (SELECT appid,recommendationid,author_steamid,voted_up FROM read_parquet({quote(original)})
             EXCEPT ALL SELECT appid,recommendationid,author_steamid,voted_up FROM read_parquet({quote(out)})))''').fetchone()[0]
        if bad or rows!=info['rows']:raise ValueError('Extra features do not match frozen review identities')
    finally:con.close()
    write_json(out.with_suffix('.json'),{'status':'complete','sha256':digest(out),'rows':rows,
        'columns':columns,'identity_multiset_differences':bad,'source_commit':source['commit'],
        'source_url':info['source_url'],'http_bytes_received':remote.bytes_received,
        'source_whole_file_sha256_independently_verified':False,'code_sha256':digest(__file__)})
    return out


def query_language(cohort):
    path=ROOT/'query_context'/f'{cohort["name"]}_languages.parquet'
    if not path.exists():
        con=duckdb.connect(str(EDA/'data/processed/kieran_original/eda_work.duckdb'),read_only=True)
        try:
            con.execute('SET threads=2');con.execute("SET memory_limit='1500MB'")
            query=SPLIT/cohort['name']/'query.parquet'
            if cohort['name']=='validation':
                join=f'''SELECT q.user_key,r.language FROM read_parquet({quote(query)}) q
                    JOIN read_parquet({quote(SPLIT/'validation/review_lineage.parquet')}) l USING(user_key,app_id,positive)
                    JOIN valid_raw r USING(user_key,app_id,positive,review_key) WHERE q.positive'''
            else:
                features=fetch_frank_features()
                join=f'''SELECT DISTINCT q.user_key,q.app_id,f.language FROM read_parquet({quote(query)}) q
                    JOIN read_parquet({quote(features)}) f
                    ON q.user_key=try_cast(f.author_steamid AS UBIGINT) AND q.app_id=f.appid AND q.positive=f.voted_up WHERE q.positive'''
            con.execute(f'''COPY (SELECT user_key,lower(language) AS "language" FROM ({join})
                WHERE language IS NOT NULL GROUP BY user_key,language
                QUALIFY row_number() OVER(PARTITION BY user_key ORDER BY count(*) DESC,lower(language) ASC)=1
                ORDER BY user_key) TO {quote(path)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        finally:con.close()
    d=pq.read_table(path).to_pydict()
    result=np.full(len(cohort['user_ids']),'',dtype=object)
    pos=np.searchsorted(cohort['user_ids'],np.asarray(d['user_key'],dtype=np.uint64))
    result[pos]=np.asarray(d['language'],dtype=object)
    return result


def metadata_features(cohort):
    apps,linux,_=catalog()
    games=pq.read_table(EDA/'legacy/data/processed/A_games.parquet',columns=['app_id','date_release']).to_pydict()
    years={int(a):(d.year if d else np.nan) for a,d in zip(games['app_id'],games['date_release'])}
    year=np.array([years.get(int(a),np.nan) for a in apps])
    present=np.isfinite(year)
    denominator=np.asarray(cohort['positive']@present.astype(float)).ravel()
    queryyear=np.asarray(cohort['positive']@np.nan_to_num(year)).ravel()/np.maximum(denominator,1)
    queryyear[denominator==0]=np.nan
    c=pq.read_table(EDA/'legacy/data/processed/C_games_2025.parquet',columns=['appid','supported_languages']).to_pydict()
    languages={int(a):str(v or '').lower() for a,v in zip(c['appid'],c['supported_languages'])}
    texts=[languages.get(int(a),'') for a in apps[linux]]
    userlang=query_language(cohort)
    aliases={'schinese':'simplified chinese','tchinese':'traditional chinese','koreana':'korean',
        'latam':'spanish - latin america','brazilian':'portuguese - brazil','ukrainian':'ukrainian'}
    compatibility={lang:np.array([float(aliases.get(lang,lang) in text) if text else -1. for text in texts],dtype=np.float32)
        for lang in np.unique(userlang) if lang}
    return year[linux],queryyear,userlang,compatibility


if __name__=='__main__':print(fetch_frank_features())
