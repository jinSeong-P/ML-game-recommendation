"""Keep v2 description settings while filling the same ten audited gaps."""
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from eda_large_review_frank import EDA,digest
from tfidf_content import clean_description,tokenize_description
from experiment_v3_common import ROOT,PROTOCOL,write_json,model_files
from evaluate_baselines import inventory


def build():
    old=ROOT/'content/description_missing_fill'
    source=json.loads((old/'manifest.json').read_text('utf-8'))
    folder=ROOT/'content_models/description_missing_fill_bigram'
    if folder.exists():return folder
    plan=ROOT/'filled_bigram_protocol.json'
    settings={'status':'fixed_before_filled_bigram_metrics','parent_sha256':digest(PROTOCOL),
        'fill_source':model_files(old),'ngram_range':[1,2],'min_df':1,'max_df':1.,
        'sublinear_tf':False,'metadata_weight':.5,'description_weight':.5,
        'source_rule':'Exact ten audited replacements; all other v1 short descriptions unchanged.',
        'reason':'Compare missing-content fill while preserving the selected v2 bigram feature settings.',
        'code_sha256':digest(__file__)}
    if plan.exists() and json.loads(plan.read_text('utf-8'))!=settings:raise ValueError('Filled-bigram plan changed')
    write_json(plan,settings)
    baseline=EDA/'data/processed/modeling/tfidf_content_v1'
    d=pq.read_table(baseline/'descriptions.parquet',columns=['app_id','description_clean']).to_pydict()
    texts=list(d['description_clean']);apps=np.asarray(d['app_id'])
    games=pq.read_table(ROOT/'content/sources/frank_games.parquet',columns=['appid','about_the_game']).to_pydict()
    public={int(a):t for a,t in zip(games['appid'],games['about_the_game'])}
    for row in source['source_lineage']['filled_apps']:
        app=int(row['app_id']);index=int(np.searchsorted(apps,app));text=clean_description(public[app])
        if hashlib.sha256(text.encode()).hexdigest()!=row['clean_sha256']:raise ValueError('Audited filled text differs')
        texts[index]=text
    vectorizer=TfidfVectorizer(tokenizer=tokenize_description,token_pattern=None,lowercase=False,
        ngram_range=(1,2),min_df=1,max_df=1.,sublinear_tf=False,norm='l2',dtype=np.float32)
    matrix=vectorizer.fit_transform(texts).tocsr();matrix.sort_indices()
    folder.mkdir(parents=True)
    sparse.save_npz(folder/'tfidf_csr.npz',matrix)
    pq.write_table(pq.read_table(baseline/'game_rows.parquet'),folder/'game_rows.parquet',compression='zstd')
    write_json(folder/'manifest.json',{'status':'complete','family':'content','parameters':settings,
        'index':{'shape':list(matrix.shape),'nnz':matrix.nnz},'files':inventory(folder),
        'fitted_on_review_labels':False})
    return folder


if __name__=='__main__':print(build())
