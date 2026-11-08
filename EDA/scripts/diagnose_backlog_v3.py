"""Reproduce backlog distribution and duplicate-content diagnostics explicitly.

These describe already observed cohorts/results; they neither train nor select.
Associations across cohorts are not causal effects or independent test evidence.
"""
import json
import re
import unicodedata
from collections import Counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from eda_large_review_frank import EDA,digest,quote,log
from experiment_v3_common import ROOT,SPLIT,catalog,frozen_cohort,write_json


def normalized_title(value):
    return re.sub(r'[^\w]+',' ',unicodedata.normalize('NFKC',str(value or '')).casefold()).strip()


def distribution(cohort,support):
    targets=cohort['target_ids'];_,counts=np.unique(targets,return_counts=True)
    p=counts/counts.sum();q=cohort['positive'].tocoo();inputs=support[q.col]
    return {'users':len(targets),'target_games':len(counts),
        'top10_target_game_share':float(np.sort(counts)[-10:].sum()/len(targets)),
        'effective_target_games_exp_shannon':float(np.exp(-(p*np.log(p)).sum())),
        'effective_target_games_inverse_simpson':float(1/(p@p)),
        'positive_input_train_support_median':float(np.median(inputs)),
        'positive_input_support_at_least_10000_share':float(np.mean(inputs>=10000)),
        'negative_input_pair_share':float(cohort['negative'].nnz/(cohort['positive'].nnz+cohort['negative'].nnz)),
        'users_with_negative_input_share':float(np.mean(np.diff(cohort['negative'].indptr)>0)),
        'positive_inputs_median':float(np.median(cohort['n_positive']))}


def pair_diagnostics(cohort,text,hashes,titlegroups,apps):
    q=cohort['positive'].tocoo();targetcols=np.searchsorted(apps,cohort['target_ids'])
    maximum=np.zeros(len(cohort['user_ids']));best=np.zeros(len(maximum),dtype=np.int64)
    duplicate_text=np.zeros(len(maximum),dtype=bool);duplicate_title=np.zeros(len(maximum),dtype=bool)
    # Sparse pairwise dot products in bounded chunks; no target text leaks into
    # training or query features. This is only a target-aware diagnosis.
    for begin in range(0,len(q.row),50000):
        end=min(begin+50000,len(q.row));users=q.row[begin:end];inputcols=q.col[begin:end];t=targetcols[users]
        score=np.asarray(text[t].multiply(text[inputcols]).sum(axis=1)).ravel()
        np.maximum.at(maximum,users,score)
        np.logical_or.at(duplicate_text,users,(hashes[t]==hashes[inputcols])&(hashes[t]!=''))
        np.logical_or.at(duplicate_title,users,(titlegroups[t]==titlegroups[inputcols])&(titlegroups[t]!=''))
        # Store an input attaining the reported maximum, including across chunks.
        # Similarity alone does not establish that the products are identical.
        chosen=(score>=maximum[users]) & (score>=.8)
        best[users[chosen]]=inputcols[chosen]
    return maximum,duplicate_text,duplicate_title,best


def main():
    apps,linux,support=catalog();base=EDA/'data/processed/modeling/tfidf_content_v1'
    descriptions=pq.read_table(base/'descriptions.parquet',columns=['app_id','description_clean','clean_sha256']).to_pydict()
    texts=np.asarray(descriptions['description_clean'],dtype=object)
    hashes=np.array([h if t else '' for h,t in zip(descriptions['clean_sha256'],texts)],dtype=object)
    text=sparse.load_npz(base/'tfidf_csr.npz')
    raw=pq.read_table(EDA/'legacy/data/processed/A_games.parquet',columns=['app_id','title']).to_pydict()
    titles={int(a):str(t or '') for a,t in zip(raw['app_id'],raw['title'])}
    titlegroups=np.array([normalized_title(titles.get(int(a),'')) for a in apps],dtype=object)
    result={'status':'complete','purpose':'Observed-cohort descriptive audit, never selection evidence',
        'code_sha256':digest(__file__),'protocol_sha256':digest(EDA/'data/reference/improvement_protocol_v3.json'),
        'definitions':{'same_title':'NFKC/casefold/punctuation-normalized exact title; not proof of identical product',
        'same_description':'Exact nonempty cleaned baseline description SHA256',
        'high_similarity':'Maximum single-positive-input unigram baseline description cosine >= threshold',
        'support':'Distinct positive train users in frozen catalog; input statistics weight input pairs, not users'},'cohorts':{}}
    private=ROOT/'diagnostics';private.mkdir(parents=True,exist_ok=True)
    for name in ['validation','test']:
        c=frozen_cohort(name);d=distribution(c,support)
        maximum,sametext,sametitle,best=pair_diagnostics(c,text,hashes,titlegroups,apps)
        paths={kind:EDA/'data/processed/modeling/baseline_v1'/name/kind/'per_user.parquet' for kind in ['item_knn','tfidf']}
        outcomes={kind:pq.read_table(p).to_pydict() for kind,p in paths.items()}
        group=(support[np.searchsorted(apps,c['target_ids'])]>=100)&(support[np.searchsorted(apps,c['target_ids'])]<1000)
        def stats(mask):
            n=int(mask.sum());value={'users':n,'fraction':n/len(mask)}
            for kind,per in outcomes.items():
                hits=np.asarray(per['target_rank'])>0
                value[kind]={'Recall@10':float(hits[mask].mean()) if n else None,
                    'share_of_all_model_hits':float((hits&mask).sum()/max(hits.sum(),1))}
            return value
        d['support100_999']={'users':int(group.sum()),'positive_inputs_median':float(np.median(c['n_positive'][group])),
            'high_similarity_at_least_0_2_share':float(np.mean(maximum[group]>=.2)),
            'item_knn_Recall@10':float(np.mean(np.asarray(outcomes['item_knn']['target_rank'])[group]>0))}
        d['similarity_subgroups']={str(threshold):stats(maximum>=threshold) for threshold in [.2,.8]}
        d['exact_clean_description_overlap']=stats(sametext)
        d['exact_normalized_title_overlap']=stats(sametitle)
        d['without_cosine_at_least_0_8']=stats(maximum<.8)
        cases=np.flatnonzero(maximum>=.8)
        pq.write_table(pa.table({'user_key':c['user_ids'][cases],'target_app_id':c['target_ids'][cases],
            'input_app_id':apps[best[cases]],'target_title':[titles.get(int(a),'') for a in c['target_ids'][cases]],
            'input_title':[titles.get(int(a),'') for a in apps[best[cases]]],
            'exact_description':sametext[cases],'exact_title':sametitle[cases],'maximum_cosine':maximum[cases]}),private/f'{name}_similar_content_cases.parquet',compression='zstd')
        result['cohorts'][name]=d
        log('Backlog descriptive diagnostics complete: '+name)
    con=duckdb.connect()
    try:
        con.execute('SET threads=2');con.execute("SET memory_limit='1500MB'")
        source=SPLIT/'train/positive_catalog.parquet'
        activity=con.execute(f'''WITH degree AS (SELECT user_key,count(*) n FROM read_parquet({quote(source)}) GROUP BY user_key)
            SELECT count(*),count(*) FILTER(WHERE n>=25),sum(n*(n-1)/2),sum(n*(n-1)/2) FILTER(WHERE n>=25) FROM degree''').fetchone()
        result['train_reviewer_activity']={'users':activity[0],'users_with_at_least25':activity[1],
            'user_fraction_at_least25':activity[1]/activity[0],
            'unordered_item_pair_contribution_fraction_from_at_least25':activity[3]/activity[2]}
        owner=EDA/'data/processed/frank/user_games_identity_columns.parquet'
        targetusers=SPLIT/'test/users.parquet'
        coverage=con.execute(f'''SELECT count(DISTINCT u.user_key) FROM read_parquet({quote(targetusers)}) u
            JOIN read_parquet({quote(owner)}) g ON u.user_key=try_cast(g.steamid AS UBIGINT)''').fetchone()[0]
        result['frank_user_games_coverage']={'covered_primary_test_users':coverage,'total_primary_test_users':24342,
            'interpretation':'Coverage audit; use review playtime extra projection rather than generalizing these ownership rows to uncovered users.'}
    finally:con.close()
    result['limits']=['Observed cohort associations do not establish why frank is better in a support band.',
        'High content similarity is not proof of a series or duplicate game; reviewed case titles stay in private processed data.',
        'Do not exclude high-similarity users from official metrics or choose settings from these frank diagnostics.']
    write_json(EDA/'reports/backlog_diagnostics_v3.json',result)


if __name__=='__main__':main()
