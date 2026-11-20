"""Account-disjoint Kieran retraining and controlled Forgemaster cap study.

Stages: prepare, content, source, fit, evaluate, report. All account-level and
model artifacts are isolated under ignored forgemaster_cap_study_v1. Historical
split_v1, model locks, reports and official results are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from build_modeling_splits import create_holdout, validate_holdout
from eda_large_review_frank import build_source, digest, log, one
from evaluate_baselines import rank_positive
from item_knn import _build_user_item_csr

EDA=Path(__file__).resolve().parents[1]
ROOT=EDA / "data/processed/modeling/forgemaster_cap_study_v1"
PROTOCOL=EDA / "data/reference/forgemaster_cap_protocol_v1.json"
PAIRS=EDA / "data/processed/forgemaster_v1/clean_pairs.parquet"
FRANK=EDA / "data/processed/forgemaster_eda/frank_reviews_identity.parquet"
CAT=EDA / "data/reference/catalog.parquet"
SEEDS=[20261108,20261109,20261110,20261111,20261112]
MODEL_NAMES=['binary_item_knn_k100','activity_inverse_degree_cosine_k100','metadata_extended_tfidf','train_popularity']


def q(value):
    return "'"+str(value).replace('\\','/').replace("'","''")+"'"


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item())+'\n',encoding='utf-8')
    temp.replace(path)


def connect():
    ROOT.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(ROOT / 'study.duckdb'))
    con.execute("SET threads=2")
    con.execute("SET memory_limit='5GB'")
    con.execute(f"SET temp_directory={q(ROOT / 'duckdb_temp')}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"CREATE OR REPLACE VIEW catalog AS SELECT * FROM read_parquet({q(CAT)})")
    return con


def inventory(paths):
    return {str(path.relative_to(ROOT)).replace('\\','/'):{'bytes':path.stat().st_size,'sha256':digest(path)} for path in paths}


def verify_stage(path):
    if not path.exists():
        return None
    m=json.loads(path.read_text('utf-8'))
    if m.get('protocol_sha256')!=digest(PROTOCOL) or m.get('status')!='complete':
        raise ValueError(f'Incompatible/incomplete existing stage: {path}')
    for name,info in m.get('files',{}).items():
        p=ROOT / name
        if not p.exists() or p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
            raise ValueError(f'Artifact changed: {p}')
    return m


def catalog_arrays():
    frame=pq.read_table(CAT).to_pandas()
    frame=frame[frame.reviewed].sort_values('app_id')
    apps=frame.app_id.to_numpy(dtype=np.int64)
    linux=frame.linux.to_numpy(dtype=bool)
    if len(apps)!=37610 or linux.sum()!=6630:
        raise ValueError('Fixed catalog dimensions changed')
    return apps,linux


def export(con,sql,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    con.execute(f'COPY ({sql}) TO {q(path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)')


def export_cohort(con,prefix,folder,full_query=False):
    folder=ROOT / 'cohorts' / folder
    export(con,f'SELECT * FROM {prefix}_cohort ORDER BY user_key',folder / 'users.parquet')
    export(con,f'SELECT * FROM {prefix}_targets ORDER BY user_key',folder / 'targets.parquet')
    export(con,f'SELECT user_key,app_id,positive,linux FROM {prefix}_query ORDER BY user_key,app_id',folder / 'query.parquet')
    if full_query:
        export(con,f'''SELECT h.user_key,h.app_id,h.positive,h.linux FROM forge_hist h
            SEMI JOIN {prefix}_cohort USING(user_key)
            ANTI JOIN {prefix}_targets USING(user_key,app_id)
            ORDER BY user_key,app_id''',folder / 'full_query.parquet')
    return list(folder.glob('*.parquet'))


def prepare():
    if verify_stage(ROOT / 'prepared.json'):
        log('Reused verified prepared cohorts');return
    con=connect()
    con.execute(f'''CREATE OR REPLACE VIEW forge_pairs AS SELECT * FROM read_parquet({q(PAIRS)})''')
    con.execute('''CREATE OR REPLACE TABLE forge_hist AS SELECT cast(steamid AS UBIGINT) user_key,
        f.app_id,f.positive,c.linux FROM forge_pairs f JOIN catalog c USING(app_id) WHERE reviewed''')
    create_holdout(con,'forge_hist','full',20261108)
    checks={'full':validate_holdout(con,'full')}
    count=con.execute('SELECT count(*) FROM full_cohort').fetchone()[0]
    if count!=395421:
        raise ValueError(f'Full EDA eligible count changed: {count}')
    paths=export_cohort(con,'full','full')
    cap_counts=[]
    for seed in SEEDS:
        prefix=f'cap{seed}'
        log(f'Preparing capped/same-target histories for seed {seed}')
        con.execute(f'''CREATE OR REPLACE TEMP TABLE cap_pairs AS SELECT * FROM forge_pairs
            QUALIFY row_number() OVER(PARTITION BY app_id ORDER BY hash(steamid,app_id,{seed}),steamid)<=100''')
        con.execute('''CREATE OR REPLACE TEMP TABLE cap_hist AS SELECT cast(steamid AS UBIGINT) user_key,
            f.app_id,f.positive,c.linux FROM cap_pairs f JOIN catalog c USING(app_id) WHERE reviewed''')
        create_holdout(con,'cap_hist',prefix,20261108)
        checks[prefix]=validate_holdout(con,prefix)
        n=con.execute(f'SELECT count(*) FROM {prefix}_cohort').fetchone()[0]
        if con.execute(f'SELECT count(*) FROM {prefix}_cohort ANTI JOIN full_cohort USING(user_key)').fetchone()[0]:
            raise AssertionError('A capped eligible user was not reserved as a full eligible user')
        cap_counts.append({'seed':seed,'users':n,'pairs':con.execute('SELECT count(*) FROM cap_pairs').fetchone()[0]})
        paths+=export_cohort(con,prefix,prefix,full_query=True)
    con.execute(f'''CREATE OR REPLACE TABLE frank_raw AS SELECT cast(author_steamid AS UBIGINT) user_key,
        cast(appid AS BIGINT) app_id,cast(recommendationid AS UBIGINT) review_key,
        voted_up positive,timestamp_created,timestamp_updated FROM read_parquet({q(FRANK)})''')
    con.execute('''CREATE OR REPLACE TABLE frank_clean AS SELECT user_key,app_id,bool_or(positive) positive
        FROM frank_raw GROUP BY user_key,app_id HAVING bool_or(positive)=bool_and(positive)''')
    con.execute('''CREATE OR REPLACE TABLE frank_reserved AS SELECT user_key FROM frank_clean
        JOIN catalog USING(app_id) WHERE reviewed GROUP BY user_key
        HAVING count(*)>=2 AND count(*) FILTER(WHERE positive AND linux)>0''')
    frank_n=con.execute('SELECT count(*) FROM frank_reserved').fetchone()[0]
    if frank_n!=87573:
        raise AssertionError(f'Frank reservation changed: {frank_n}')
    con.execute('''CREATE OR REPLACE TABLE reserved AS SELECT user_key FROM full_cohort
        UNION SELECT user_key FROM frank_reserved''')
    reserved=ROOT / 'reserved_accounts.parquet'
    export(con,'SELECT * FROM reserved ORDER BY user_key',reserved);paths.append(reserved)
    result={'status':'complete','protocol_sha256':digest(PROTOCOL),'full_users':count,
        'cap_counts':cap_counts,'frank_reserved':frank_n,
        'combined_reserved':con.execute('SELECT count(*) FROM reserved').fetchone()[0],
        'holdout_checks':checks,'sources':{str(p.relative_to(EDA)):digest(p) for p in [PAIRS,FRANK,CAT]},
        'files':inventory(paths)}
    save_json(ROOT / 'prepared.json',result);con.close()
    log('Prepared all full/capped cohorts and exact-account reservations')


def source():
    if verify_stage(ROOT / 'training_source.json'):
        log('Reused verified training source');return
    verify_stage(ROOT / 'prepared.json') or (_ for _ in ()).throw(ValueError('Prepare cohorts first'))
    folder=EDA / 'data/processed/kieran_original'
    ingestion=json.loads((folder / 'ingestion_manifest.json').read_text('utf-8'))
    if ingestion['status']!='complete' or ingestion['archive_sha256']!='ec3234d1ff0831f5a20752059df1c85dfe66ab0efde95b886c7cc3a11d0430b7':
        raise ValueError('Restored Kieran source is not the complete pinned archive')
    con=connect()
    con.execute(f"CREATE OR REPLACE TEMP VIEW full_cohort AS SELECT * FROM read_parquet({q(ROOT / 'cohorts/full/users.parquet')})")
    log('Rebuilding exact original-review identity and user/game quarantine from full Kieran source')
    quality=build_source(con,str(folder / 'reviews_*.parquet').replace('\\','/'))
    full_clean=con.execute('SELECT count(*) FROM clean').fetchone()[0]
    if full_clean!=113195034:
        raise ValueError(f'Canonical clean count differs from original audit: {full_clean}')
    audit=one(con,'''SELECT
        (SELECT count(*) FROM (SELECT DISTINCT user_key FROM clean) u SEMI JOIN full_cohort USING(user_key)) forge_test_accounts_in_kieran,
        (SELECT count(*) FROM clean SEMI JOIN full_cohort USING(user_key)) forge_test_pairs_removed_from_kieran,
        (SELECT count(*) FROM (SELECT DISTINCT user_key FROM clean) u SEMI JOIN reserved USING(user_key)) reserved_accounts_in_kieran,
        (SELECT count(*) FROM clean c JOIN catalog USING(app_id) ANTI JOIN frank_reserved USING(user_key)
            WHERE positive AND reviewed AND cast('0x'||substr(md5('steam-eda-validation-v1:'||cast(c.user_key AS VARCHAR)),1,15) AS UBIGINT)%10<>0) reconstructed_original_positive_train_pairs,
        (SELECT count(DISTINCT c.user_key) FROM clean c JOIN catalog USING(app_id)
            SEMI JOIN full_cohort USING(user_key) ANTI JOIN frank_reserved USING(user_key)
            WHERE positive AND reviewed AND cast('0x'||substr(md5('steam-eda-validation-v1:'||cast(c.user_key AS VARCHAR)),1,15) AS UBIGINT)%10<>0) forge_accounts_in_reconstructed_original_positive_train''')
    if audit['reconstructed_original_positive_train_pairs']!=78315268:
        raise ValueError('Restored original logical train partition does not match the frozen 78,315,268 pairs')
    con.execute('''CREATE OR REPLACE TABLE train_positive AS SELECT c.user_key,c.app_id FROM clean c
        JOIN catalog USING(app_id) ANTI JOIN reserved USING(user_key)
        WHERE positive AND reviewed AND
        cast('0x'||substr(md5('steam-eda-validation-v1:'||cast(c.user_key AS VARCHAR)),1,15) AS UBIGINT)%10<>0''')
    stats=one(con,'''SELECT count(*) pairs,count(DISTINCT user_key) users,count(DISTINCT app_id) apps FROM train_positive''')
    leakage=one(con,'''SELECT
        (SELECT count(*) FROM train_positive SEMI JOIN reserved USING(user_key)) train_reserved_pair_overlap,
        (SELECT count(*) FROM train_positive WHERE cast('0x'||substr(md5('steam-eda-validation-v1:'||cast(user_key AS VARCHAR)),1,15) AS UBIGINT)%10=0) train_validation_pair_overlap,
        (SELECT count(*)-count(DISTINCT (user_key,app_id)) FROM train_positive) duplicate_train_pairs''')
    if any(leakage.values()):
        raise AssertionError(leakage)
    train=ROOT / 'train_positive_catalog.parquet'
    export(con,'SELECT * FROM train_positive ORDER BY user_key,app_id',train)
    result={'status':'complete','protocol_sha256':digest(PROTOCOL),'training':stats,'raw_quality':quality,
        'identity_audit':audit,'leakage_checks':leakage,'clean_kieran_pairs':full_clean,
        'kieran_ingestion_manifest_sha256':digest(folder / 'ingestion_manifest.json'),
        'files':inventory([train])}
    save_json(ROOT / 'training_source.json',result);con.close()
    log(f'Wrote {stats["pairs"]:,} train-positive pairs from {stats["users"]:,} disjoint accounts')


def top_neighbors(cols,values,own,apps):
    keep=(cols!=own)&np.isfinite(values)&(values>0)
    cols,values=cols[keep],values[keep]
    order=np.lexsort((apps[cols],-values))[:100]
    order=order[np.argsort(cols[order],kind='stable')]
    return cols[order],values[order]


def fit():
    existing=verify_stage(ROOT / 'model_manifest.json')
    if existing:
        log('Reused verified collaborative models');return
    train=verify_stage(ROOT / 'training_source.json')
    if not train:
        raise ValueError('Restore training source first')
    apps,linux=catalog_arrays();candidate_cols=np.flatnonzero(linux)
    pf=pq.ParquetFile(ROOT / 'train_positive_catalog.parquet')
    log('Constructing binary positive train user/item CSR (singletons retained)')
    x,support,n_users=_build_user_item_csr(pf.iter_batches(batch_size=131072),apps,train['training']['pairs'],train['training']['users'])
    degree=np.diff(x.indptr).astype(np.float32)
    xcsc=x.tocsc().astype(np.float32);del x;gc.collect()
    model_root=ROOT / 'models';model_root.mkdir(exist_ok=True)
    np.save(model_root / 'positive_support.npy',support)
    paths=[model_root / 'positive_support.npy']
    for kind in ['binary_item_knn_k100','activity_inverse_degree_cosine_k100']:
        matrix_path=model_root / (kind+'.npz')
        if matrix_path.exists():
            log(f'Reusing completed matrix {kind}');paths.append(matrix_path);continue
        log(f'Fitting fixed {kind}')
        weighted=xcsc.copy()
        if kind.startswith('activity'):
            weighted.data=np.sqrt(1.0/degree[weighted.indices]).astype(np.float32)
        marginal=np.asarray(weighted.power(2).sum(axis=0)).ravel().astype(np.float64)
        xt=weighted.T.tocsr()
        col_rows=[];val_rows=[]
        for begin in range(0,len(candidate_cols),48):
            stop=min(begin+48,len(candidate_cols))
            cooc=(xt[candidate_cols[begin:stop]]@weighted).tocsr()
            for local,own in enumerate(candidate_cols[begin:stop]):
                start,end=cooc.indptr[local:local+2]
                cols=cooc.indices[start:end]
                counts=cooc.data[start:end].astype(np.float64)
                denom=np.sqrt(marginal[own]*marginal[cols])
                values=np.divide(counts,denom,out=np.zeros_like(counts),where=denom>0)
                out_cols,out_vals=top_neighbors(cols,values,own,apps)
                col_rows.append(out_cols);val_rows.append(out_vals)
            if begin%480==0:
                log(f'{kind}: candidate rows {stop:,}/{len(candidate_cols):,}')
        sizes=np.array([len(row) for row in col_rows],dtype=np.int64)
        matrix=sparse.csr_matrix((np.concatenate(val_rows),np.concatenate(col_rows),np.r_[0,np.cumsum(sizes)]),shape=(len(candidate_cols),len(apps)))
        if matrix.nnz and (not np.all(np.isfinite(matrix.data)) or np.any(matrix.data<=0)):
            raise AssertionError('Invalid fitted similarities')
        sparse.save_npz(matrix_path,matrix);paths.append(matrix_path)
        del matrix,weighted,xt,col_rows,val_rows,cooc;gc.collect()
    save_json(ROOT / 'model_manifest.json',{'status':'complete','protocol_sha256':digest(PROTOCOL),
        'train_source_manifest_sha256':digest(ROOT / 'training_source.json'),'users':n_users,
        'model_names':MODEL_NAMES[:2],'files':inventory(paths),'training_only':True})
    log('Completed both fixed collaborative models')


def content():
    if verify_stage(ROOT / 'content_manifest.json'):
        log('Reused verified content model');return
    from tfidf_content import select_description,clean_description
    from improvement_tfidf import parse_metadata_labels,choose_tag_source,normalize_concept
    from experiment_content_v3 import make_word_vectorizer,make_metadata_vectorizer,_extra_metadata_features
    ROOT.mkdir(parents=True,exist_ok=True)
    a=EDA / 'data/processed/A_games_metadata.parquet'
    c=EDA / 'data/processed/C_games_2025.parquet'
    old={r['file']:r['sha256'] for r in json.loads((EDA / 'data/reference/manifest.json').read_text('utf-8'))['former_catalog_inputs']}
    if digest(a)!=old[a.name] or digest(c)!=old[c.name]:
        raise ValueError('Public metadata source fingerprints differ from the fixed original catalog')
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1GB'")
    rows=con.execute(f'''SELECT f.app_id,f.linux,a.description,a.tags a_tags,
        c.short_description,c.tags c_tags,c.genres,c.categories,c.developers,c.publishers
        FROM read_parquet({q(CAT)}) f LEFT JOIN read_parquet({q(a)}) a USING(app_id)
        LEFT JOIN read_parquet({q(c)}) c ON f.app_id=c.appid WHERE f.reviewed ORDER BY f.app_id''').fetch_arrow_table().to_pylist()
    con.close()
    apps,linux=catalog_arrays()
    if [r['app_id'] for r in rows]!=apps.tolist():
        raise ValueError('Content mapping differs from fixed input catalog')
    documents=[];metadata=[];audit=[]
    for row in rows:
        raw,origin=select_description(row['description'],row['short_description'])
        text=clean_description(raw);documents.append(text or '')
        al,ast,_=parse_metadata_labels(row['a_tags'],'tags')
        cl,cst,_=parse_metadata_labels(row['c_tags'],'tags')
        chosen,tag_source=choose_tag_source(al,ast,cl,cst)
        genres,_,_=parse_metadata_labels(row['genres'],'genres')
        features={f'tag::{v}' for label in chosen if (v:=normalize_concept(label))}
        features|={f'genre::{v}' for label in genres if (v:=normalize_concept(label))}
        extra,_=_extra_metadata_features(row)
        features=sorted(features|set(extra));metadata.append(features)
        audit.append({'app_id':row['app_id'],'linux':row['linux'],'description_source':origin,
            'tag_source':tag_source,'description_clean':text,'metadata_features':features})
    log('Fitting fixed public-catalog description bigram and extended metadata TF-IDF')
    description=make_word_vectorizer((1,2)).fit_transform(documents).tocsr()
    features=make_metadata_vectorizer().fit_transform(metadata).tocsr()
    folder=ROOT / 'models';folder.mkdir(exist_ok=True)
    dp,mp=folder / 'description_csr.npz',folder / 'metadata_csr.npz'
    sparse.save_npz(dp,description);sparse.save_npz(mp,features)
    ap=folder / 'content_source_audit.parquet';pq.write_table(pa.Table.from_pylist(audit),ap,compression='zstd')
    save_json(ROOT / 'content_manifest.json',{'status':'complete','protocol_sha256':digest(PROTOCOL),
        'model':'metadata_extended_tfidf','parameters':{'description_weight':0.5,'metadata_weight':0.5,'description_ngram':[1,2],
        'scoring':'separate cosine of equal-weight positive-input mean profiles'},
        'description_shape':description.shape,'metadata_shape':features.shape,
        'description_nonzero_rows':int(np.count_nonzero(np.diff(description.indptr))),
        'metadata_nonzero_rows':int(np.count_nonzero(np.diff(features.indptr))),
        'sources':{str(p.relative_to(EDA)):digest(p) for p in [a,c,CAT]},'uses_review_outcomes':False,
        'files':inventory([dp,mp,ap])})
    log('Completed fixed content model')


def positions(values,reference):
    values=np.asarray(values)
    pos=np.searchsorted(reference,values)
    if np.any(pos>=len(reference)) or np.any(reference[pos]!=values):
        raise ValueError('Unknown account or App ID in saved cohort')
    return pos


def load_cohort(folder,query_file='query.parquet',mask_file=None):
    apps,linux=catalog_arrays();candidates=apps[linux]
    targets=pq.read_table(folder / 'targets.parquet').to_pandas()
    users=targets.user_key.to_numpy(dtype=np.uint64)
    if np.any(users[1:]<=users[:-1]):
        raise ValueError('Targets are not unique sorted exact account IDs')
    query=pq.read_table(folder / query_file).to_pandas()
    r=positions(query.user_key.to_numpy(dtype=np.uint64),users)
    c=positions(query.app_id.to_numpy(dtype=np.int64),apps)
    positive=query.positive.to_numpy(dtype=bool)
    shape=(len(users),len(apps))
    full=sparse.coo_matrix((np.ones(len(r),dtype=np.float32),(r,c)),shape=shape).tocsr()
    pos=sparse.coo_matrix((np.ones(int(positive.sum()),dtype=np.float32),(r[positive],c[positive])),shape=shape).tocsr()
    if full.nnz!=len(query):
        raise AssertionError('Duplicate saved query pair')
    if mask_file and mask_file!=query_file:
        mask_query=pq.read_table(folder / mask_file).to_pandas()
        mr=positions(mask_query.user_key.to_numpy(dtype=np.uint64),users)
        mc=positions(mask_query.app_id.to_numpy(dtype=np.int64),apps)
        mask=sparse.coo_matrix((np.ones(len(mr),dtype=np.float32),(mr,mc)),shape=shape).tocsr()[:,linux]
    else:
        mask=full[:,linux]
    target_ids=targets.app_id.to_numpy(dtype=np.int64)
    target_candidates=positions(target_ids,candidates)
    if np.any(np.asarray(mask[np.arange(len(users)),target_candidates]).ravel()):
        raise AssertionError('Hidden target was in exclusions')
    if np.any(np.asarray(full[np.arange(len(users)),positions(target_ids,apps)]).ravel()):
        raise AssertionError('Hidden target was in query')
    n=np.diff(full.indptr);p=np.diff(pos.indptr)
    if np.any(n<5) or np.any(p<1) or not targets.positive.all() or not targets.linux.all():
        raise AssertionError('Ineligible saved cohort')
    return {'users':users,'positive':pos,'mask':mask.tocsr(),'target_ids':target_ids,
        'target_candidates':target_candidates,'n_total':n,'n_positive':p}


def wilson(hits,n):
    if not n:return [None,None]
    z=1.959963984540054;rate=hits/n
    center=(rate+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def metric_summary(frame):
    n=len(frame);hits=int((frame.hit_rank>0).sum())
    macro=frame.groupby('target_app_id')[['recall','ndcg']].mean().mean()
    return {'users':n,'hits':hits,'Recall@10':hits/n,'NDCG@10':float(frame.ndcg.mean()),
        'Recall@10_wilson95':wilson(hits,n),
        'empty_recommendation_rate':float((frame.n_recommendations==0).mean()),
        'mean_recommendation_count':float(frame.n_recommendations.mean()),
        'mean_input_count':float(frame.n_total_inputs.mean()),
        'mean_positive_input_count':float(frame.n_positive_inputs.mean()),
        'target_games':int(frame.target_app_id.nunique()),
        'game_macro_Recall@10':float(macro['recall']),'game_macro_NDCG@10':float(macro['ndcg']),
        'unseen_target_rate':float((frame.target_train_support==0).mean()),
        'target_support_bands':{label:{'users':len(group),'Recall@10':float(group.recall.mean()),'NDCG@10':float(group.ndcg.mean())}
            for label,group in frame.groupby(pd.cut(frame.target_train_support,[-1,0,99,999,9999,np.inf],
                labels=['zero','1-99','100-999','1000-9999','10000+']),observed=True)}}


def score_model(name,query,matrices,support,linux):
    n=np.maximum(np.asarray(query.sum(axis=1)).ravel(),1)
    if name=='train_popularity':
        return np.broadcast_to(support[linux],(query.shape[0],int(linux.sum()))).astype(np.float64,copy=True)
    if name=='metadata_extended_tfidf':
        output=None
        for feature in matrices:
            profiles=normalize(query@feature,norm='l2',copy=False)
            scores=(profiles@feature[linux].T).toarray()
            output=.5*scores if output is None else output+.5*scores
        return output
    return (query@matrices[0].T).toarray()/n[:,None]


def evaluate_case(case,name,cohort,matrices,support,apps,linux):
    folder=ROOT / 'evaluations' / case / name
    if (folder / 'manifest.json').exists():
        m=verify_stage(folder / 'manifest.json')
        return m['metrics']
    folder.mkdir(parents=True,exist_ok=True)
    candidates=apps[linux];n=len(cohort['users'])
    ranks=np.zeros(n,dtype=np.int8);lengths=np.zeros(n,dtype=np.int8)
    target_scores=np.zeros(n,dtype=np.float64)
    schema=pa.schema([('user_key',pa.uint64()),('app_id',pa.int64()),('rank',pa.int8()),('score',pa.float64())])
    log(f'Evaluating {case}/{name}: {n:,} users')
    with pq.ParquetWriter(folder / 'recommendations.parquet',schema,compression='zstd') as writer:
        for start in range(0,n,256):
            stop=min(start+256,n)
            scores=score_model(name,cohort['positive'][start:stop],matrices,support,linux)
            mr,mc=cohort['mask'][start:stop].nonzero();scores[mr,mc]=0
            target_scores[start:stop]=scores[np.arange(stop-start),cohort['target_candidates'][start:stop]]
            out_users=[];out_apps=[];out_rank=[];out_scores=[]
            for row,values in enumerate(scores):
                idx=start+row
                selected=rank_positive(values,candidates)
                lengths[idx]=len(selected)
                hit=np.flatnonzero(selected==cohort['target_candidates'][idx])
                if len(hit):ranks[idx]=int(hit[0])+1
                out_users.extend([int(cohort['users'][idx])]*len(selected));out_apps.extend(candidates[selected].tolist())
                out_rank.extend(range(1,len(selected)+1));out_scores.extend(values[selected].astype(float).tolist())
            writer.write_table(pa.Table.from_pydict({'user_key':out_users,'app_id':out_apps,'rank':out_rank,'score':out_scores},schema=schema))
            if start and start%51200==0:log(f'{case}/{name}: {start:,}/{n:,}')
    hit=ranks>0;ndcg=np.zeros(n,dtype=float)
    ndcg[hit]=1.0/np.log2(ranks[hit].astype(np.float64)+1.0)
    frame=pd.DataFrame({'user_key':cohort['users'],'target_app_id':cohort['target_ids'],
        'hit_rank':ranks,'n_recommendations':lengths,'recall':hit.astype(float),'ndcg':ndcg,
        'n_total_inputs':cohort['n_total'],'n_positive_inputs':cohort['n_positive'],
        'target_score':target_scores,'target_train_support':support[positions(cohort['target_ids'],apps)]})
    per_user=folder / 'per_user.parquet';pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),per_user,compression='zstd')
    summary=metric_summary(frame)
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1GB'")
    qp=ROOT / 'cohorts' / ('full' if case=='full' else case.split('_')[0])
    mask_name='query.parquet' if case=='full' or case.endswith('_capped') else 'full_query.parquet'
    # Read saved recommendation data back; never trust an in-memory PASS flag.
    checks=one(con,f'''SELECT
        (SELECT count(*) FROM read_parquet({q(folder / 'recommendations.parquet')}) WHERE rank<1 OR rank>10 OR score<=0 OR NOT isfinite(score)) invalid_recommendations,
        (SELECT count(*) FROM (SELECT user_key,app_id,count(*) n FROM read_parquet({q(folder / 'recommendations.parquet')}) GROUP BY user_key,app_id HAVING n<>1)) duplicate_recommendations,
        (SELECT count(*) FROM read_parquet({q(folder / 'recommendations.parquet')}) r
            JOIN read_parquet({q(qp / mask_name)}) x USING(user_key,app_id)) observed_items_recommended,
        (SELECT count(*) FROM read_parquet({q(per_user)}) p LEFT JOIN
            (SELECT user_key,count(*) n,max(rank) last_rank FROM read_parquet({q(folder / 'recommendations.parquet')}) GROUP BY user_key) r USING(user_key)
            WHERE p.n_recommendations<>coalesce(r.n,0) OR coalesce(r.n,0)<>coalesce(r.last_rank,0)) saved_length_errors,
        (SELECT count(*) FROM read_parquet({q(per_user)}) p LEFT JOIN
            read_parquet({q(folder / 'recommendations.parquet')}) r ON p.user_key=r.user_key AND p.target_app_id=r.app_id
            WHERE p.hit_rank<>coalesce(r.rank,0)) saved_hit_rank_errors,
        (SELECT count(*) FROM read_parquet({q(per_user)})) saved_users''')
    if any(v for k,v in checks.items() if k!='saved_users') or checks['saved_users']!=n:
        raise AssertionError(checks)
    con.close()
    save_json(folder / 'manifest.json',{'status':'complete','protocol_sha256':digest(PROTOCOL),
        'case':case,'model':name,'metrics':summary,'saved_output_checks':checks,
        'evaluator_sha256':digest(Path(__file__)),
        'model_manifest_sha256':digest(ROOT / ('content_manifest.json' if name=='metadata_extended_tfidf' else 'model_manifest.json')),
        'files':inventory([per_user,folder / 'recommendations.parquet'])})
    log(f'{case}/{name}: Recall={summary["Recall@10"]:.4%}; NDCG={summary["NDCG@10"]:.5f}')
    return summary


def evaluate():
    verify_stage(ROOT / 'prepared.json') or (_ for _ in ()).throw(ValueError('Prepare first'))
    verify_stage(ROOT / 'model_manifest.json') or (_ for _ in ()).throw(ValueError('Fit collaborative models first'))
    verify_stage(ROOT / 'content_manifest.json') or (_ for _ in ()).throw(ValueError('Fit content first'))
    apps,linux=catalog_arrays();support=np.load(ROOT / 'models/positive_support.npy')
    outputs=[]
    for name in MODEL_NAMES:
        if name=='metadata_extended_tfidf':
            matrices=[sparse.load_npz(ROOT / 'models/description_csr.npz'),sparse.load_npz(ROOT / 'models/metadata_csr.npz')]
        elif name=='train_popularity':matrices=[]
        else:matrices=[sparse.load_npz(ROOT / 'models' / (name+'.npz'))]
        full=load_cohort(ROOT / 'cohorts/full')
        outputs.append({'case':'full','model':name,**evaluate_case('full',name,full,matrices,support,apps,linux)})
        del full;gc.collect()
        for seed in SEEDS:
            folder=ROOT / 'cohorts' / f'cap{seed}'
            for label,query,mask in [('full_history','full_query.parquet',None),
                ('capped_full_mask','query.parquet','full_query.parquet'),('capped','query.parquet',None)]:
                case=f'cap{seed}_{label}'
                cohort=load_cohort(folder,query,mask)
                outputs.append({'case':case,'model':name,**evaluate_case(case,name,cohort,matrices,support,apps,linux)})
                del cohort;gc.collect()
        del matrices;gc.collect()
    save_json(ROOT / 'evaluation_summary.json',{'status':'complete','protocol_sha256':digest(PROTOCOL),'results':outputs})
    log('Completed all frozen model/cohort comparisons')


def report():
    from summarize_forgemaster_model_study import main as summarize
    summarize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','source','fit','content','evaluate','report'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage]()


if __name__=='__main__':
    main()
