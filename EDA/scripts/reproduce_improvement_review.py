"""Reproduce saved-v1 review statistics and freeze source snapshots for v2."""
from pathlib import Path
import json
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from eda_large_review_frank import EDA,digest,log
from improvement_common import ROOT,V1,PROTOCOL,SPLIT,write_json,protective_summary,log_groups,LOG_LABELS
from evaluate_baselines import load_cohort,metrics

def run():
    out=ROOT/'reproduction'; out.mkdir(parents=True,exist_ok=True)
    lockpath=EDA/'data/reference/baseline_run_v1.json'
    lock=json.loads(lockpath.read_text('utf-8'))
    paths={'protocol_sha256':EDA/'data/reference/baseline_protocol_v1.json',
      'split_manifest_sha256':SPLIT/'manifest.json',
      'tfidf_manifest_sha256':EDA/'data/processed/modeling/tfidf_content_v1/manifest.json',
      'item_knn_manifest_sha256':EDA/'data/processed/modeling/item_knn_v1/manifest.json',
      'evaluator_sha256':EDA/'scripts/evaluate_baselines.py',
      'tfidf_code_sha256':EDA/'scripts/tfidf_content.py','item_knn_code_sha256':EDA/'scripts/item_knn.py',
      'item_knn_audit_sha256':EDA/'reports/item_knn_v1_audit.json'}
    checked={k:digest(p)==lock[k] for k,p in paths.items()}
    if not all(checked.values()):raise ValueError('Frozen v1 changed')
    provenance=ROOT/'provenance';provenance.mkdir(exist_ok=True)
    snapshot=provenance/'source_manifest_at_v2_start.json'
    if not snapshot.exists():shutil.copyfile(EDA/'data/source_manifest.json',snapshot)
    graph=sparse.load_npz(EDA/'data/processed/modeling/item_knn_v1/similarities_csr.npz')
    d=pq.read_table(EDA/'data/processed/modeling/item_knn_v1/game_rows.parquet').to_pydict()
    apps=np.asarray(d['app_id']); linux=np.asarray(d['linux']); support=np.asarray(d['positive_user_support'])
    candidate_cols=np.flatnonzero(linux);candidates=apps[linux]
    rows=np.repeat(np.arange(len(candidates)),np.diff(graph.indptr))
    raw=graph.data*np.sqrt(support[candidate_cols[rows]].astype(float)*support[graph.indices])
    if not np.allclose(raw,np.rint(raw),atol=1e-5,rtol=0):raise ValueError('Cannot recover integer saved-edge counts')
    cooc=np.rint(raw).astype(np.int64)
    content=EDA/'data/processed/modeling/tfidf_content_v1'
    text=pq.read_table(content/'descriptions.parquet').to_pydict()
    tokens=np.asarray(text['token_count']); cutoff=float(np.quantile(tokens[linux],.8))
    matrix=sparse.load_npz(content/'tfidf_csr.npz'); missing=apps[linux & (np.diff(matrix.indptr)==0)]
    top10=candidates[np.lexsort((candidates,-support[linux]))[:10]]
    report={'version':'baseline-review-reproduction-v2','code_sha256':digest(__file__),
      'protocol_sha256':digest(PROTOCOL),'frozen_v1_checks':checked,
      'source_manifest_snapshot_sha256':digest(snapshot),
      'source_manifest_train_sha256':json.loads((EDA/'data/processed/modeling/item_knn_v1/manifest.json').read_text('utf-8'))['sources']['source_manifest_sha256'],
      'source_manifest_note':'Training hash refers to training-time state; current completion flags are mutable. Actual pinned train files are checked by builder.',
      'saved_knn_edges':{'total':len(cooc),'one':int((cooc==1).sum()),'less_than_five':int((cooc<5).sum()),
         'one_fraction':float((cooc==1).mean()),'less_than_five_fraction':float((cooc<5).mean())},
      'textless_linux_apps':missing.tolist(),'cohorts':{},
      'unreproduced_original_definition':['Original custom 21-stopword list/formula absent: do not claim exact reconstruction. T0 uses explicitly named documented terms.',
        'Original validation cosine sample seed and sampling rule absent: no claim to reproduce 3.4% on the same sample.'],
      'token_concentration_definition':{'population':'6630 Linux candidates','p80_cutoff':cutoff,'strictly_above_cutoff_apps':int((linux & (tokens>cutoff)).sum()),'exact_top20_apps':1326,'tie_break':'App ID ascending'}}
    baseline_report=json.loads((EDA/'reports/baseline_v1.json').read_text('utf-8'))
    protocol=json.loads(PROTOCOL.read_text('utf-8'))
    for cohort in ('validation','test'):
        manifest=json.loads((V1/cohort/'manifest.json').read_text('utf-8'))
        for rel,info in manifest['files'].items():
            if digest(V1/cohort/rel)!=info['sha256']:raise ValueError('Saved v1 output changed')
        report['cohorts'][cohort]={}
        for kind in ('item_knn','tfidf'):
            folder=V1/cohort/kind
            summary=protective_summary(folder/'per_user.parquet',folder/'recommendations.parquet')
            old=baseline_report[cohort]['models'][kind]
            for field in ('Recall@10','NDCG@10'):
                if abs(summary[field]-old[field])>1e-12:raise ValueError('v1 aggregate changed')
            u=pq.read_table(folder/'per_user.parquet').to_pydict()
            tar=np.asarray(u['target_app_id']); hits=np.asarray(u['recall_at_10']); ts=np.asarray(u['target_train_support'])
            summary['top10_popular_target_share']=float(np.isin(tar,top10).mean())
            summary['top10_popular_hit_share']=float(hits[np.isin(tar,top10)].sum()/hits.sum())
            summary['support_at_least_100000_share']=float((ts>=100000).mean())
            summary['textless_target_users']=int(np.isin(tar,missing).sum())
            if kind=='tfidf':
                rec=pq.read_table(folder/'recommendations.parquet').column('app_id').to_numpy()
                recrows=np.searchsorted(apps,rec)
                ids,counts=np.unique(rec,return_counts=True); order=np.lexsort((ids,-counts))[:4]
                top20=np.flatnonzero(linux)[np.lexsort((apps[linux],-tokens[linux]))[:1326]]
                summary['recommendation_concentration']={'slots':len(rec),
                    'top_four':[{'app_id':int(ids[i]),'slots':int(counts[i]),'slot_fraction':float(counts[i]/len(rec))} for i in order],
                    'strict_p80_token_slot_share':float((tokens[recrows]>cutoff).mean()),
                    'exact_top20_token_slot_share':float(np.isin(recrows,top20).mean())}
            report['cohorts'][cohort][kind]=summary
        # Diagnostic controls use the same unmodified cohort and exclusions.
        c=load_cohort(SPLIT,cohort,apps,candidates,protocol)
        # Retrospective content-similarity strata on frozen v1 only. These never
        # filter official metrics or select v2 settings.
        maximum=np.zeros(len(c['user_ids']),dtype=np.float32)
        q=c['positive']; owners=np.repeat(np.arange(len(maximum)),np.diff(q.indptr))
        target_rows=np.searchsorted(apps,c['target_ids'])
        for start in range(0,len(owners),20000):
            stop=min(start+20000,len(owners)); who=owners[start:stop]
            similarity=np.asarray(matrix[q.indices[start:stop]].multiply(matrix[target_rows[who]]).sum(axis=1)).ravel()
            np.maximum.at(maximum,who,similarity)
        strata={'method':'max frozen description cosine between held-out target and any remaining positive query; post-hoc diagnostic only',
           'cosine_ge_0_2_users':int((maximum>=.2).sum()),'cosine_ge_0_8_users':int((maximum>=.8).sum()),'models':{}}
        for kind in ('item_knn','tfidf'):
            u=pq.read_table(V1/cohort/kind/'per_user.parquet').to_pydict()
            hits=np.asarray(u['recall_at_10']);keep=maximum<.8
            strata['models'][kind]={'hits_in_ge_0_2':int(hits[maximum>=.2].sum()),
                'share_of_hits_in_ge_0_2':float(hits[maximum>=.2].sum()/hits.sum()),
                'below_0_8_users':int(keep.sum()),'below_0_8_conditional_recall':float(hits[keep].mean())}
        a=pq.read_table(V1/cohort/'item_knn/per_user.parquet').to_pydict()
        b=pq.read_table(V1/cohort/'tfidf/per_user.parquet').to_pydict()
        ah=np.asarray(a['recall_at_10']);bh=np.asarray(b['recall_at_10']); az=np.asarray(a['target_score'])==0
        report['cohorts'][cohort]['retrospective_text_similarity']=strata
        report['cohorts'][cohort]['top10_hit_union']={'Recall@10':float(((ah>0)|(bh>0)).mean()),
            'note':'Oracle union of original top10 hits, not measured fusion performance or a bound on all possible score fusion',
            'tfidf_recall_when_knn_target_zero':float(bh[az].mean())}
        order=np.lexsort((candidates,-support[linux])); order=order[support[linux][order]>0]
        ranks=np.zeros(len(c['user_ids']),dtype=np.int8)
        for row in range(len(ranks)):
            excluded=set(c['exclusions'][row].indices)
            chosen=[i for i in order[:100] if i not in excluded][:10]
            if len(chosen)<10:chosen=[i for i in order if i not in excluded][:10]
            match=np.flatnonzero(np.asarray(chosen)==c['target_candidates'][row])
            if len(match):ranks[row]=match[0]+1
        controls=out/f'{cohort}_popularity.parquet'
        recall=(ranks>0).astype(float);ndcg=np.where(ranks>0,1/np.log2(np.maximum(ranks,1)+1),0.)
        pq.write_table(pa.table({'user_key':pa.array(c['user_ids'],type=pa.uint64()),'target_app_id':c['target_ids'],
          'n_positive_inputs':c['n_positive'],'n_total_inputs':c['n_total'],'target_train_support':support[linux][c['target_candidates']],
          'target_rank':ranks,'recommendation_count':np.full(len(ranks),10,dtype=np.int8),'recall_at_10':recall,'ndcg_at_10':ndcg}),controls,compression='zstd')
        report['cohorts'][cohort]['popularity']=protective_summary(controls)
        log('Reproduced saved v1 and popularity controls: '+cohort)
    reweighted={}
    for kind in ('item_knn','tfidf'):
        v=report['cohorts']['validation'][kind]['by_support_log'];t=report['cohorts']['test'][kind]['by_support_log']
        reweighted[kind]={'approximate_recall':sum((v[k]['Recall@10'] or 0)*t[k]['users'] for k in LOG_LABELS)/protocol['cohort_users']['test'],
           'note':'Descriptive standardization only, not causal attribution. Train-support-zero has no validation observation and is assigned 0 for this approximation. Never used to weight validation selection.'}
    report['posthoc_log_bin_standardization']=reweighted
    report['passed']=True
    write_json(EDA/'reports/baseline_review_reproduction_v2.json',report)
    write_json(out/'manifest.json',{'status':'complete','report_sha256':digest(EDA/'reports/baseline_review_reproduction_v2.json'),
      'code_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'source_snapshot_sha256':digest(snapshot)})
    return report

if __name__=='__main__':run()
