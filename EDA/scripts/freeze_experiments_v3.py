"""Freeze the finite backlog experiment plan before calculating new metrics."""
import json
from datetime import datetime,timezone

from eda_large_review_frank import EDA,digest
from improvement_common import write_json


def protocol():
    return {
        'version':'recommendation-experiments-v3','status':'fixed_before_v3_metrics',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User requested all backlog improvement ideas to be tested; document using standard terminology.',
        'split_manifest_sha256':digest(EDA/'data/processed/modeling/split_v1/manifest.json'),
        'previous_selection_sha256':digest(EDA/'data/reference/improvement_selection_v2.json'),
        'previous_protocol_sha256':digest(EDA/'data/reference/improvement_protocol_v2.json'),
        'primary_cohort':'Unchanged validation: 325639 users, one hidden positive Linux target, fixed inputs and exclusions, 6630 candidates.',
        'primary':'Whole-validation NDCG@10; Recall@10 secondary; no frank-based selection.',
        'controls':{'collaborative':'v2-retained Item-KNN K100','content':'v2 description bigram .5 + metadata .5','hybrid':'v2-retained Item-KNN K100'},
        'adoption':'Paired NDCG bootstrap 95% lower bound >0; no loss in Recall, game-macro NDCG, target support 1-999 NDCG; empty users do not increase. Compare with current same-family v2 selection. Hybrid/new collaborative versus Item-KNN control.',
        'bootstrap':{'replicates':2000,'seed':20261108,'interpretation':'Post-selection exploratory uncertainty, not independent confirmation; multiple candidates are not multiplicity-adjusted.'},
        'train_only':True,'quarantine':'unchanged','test_accounts_reserved':87573,
        'rank_rules':'Top10 strictly positive finite scores; App ID ascending ties; retain empty users in denominator; exclude all observed inputs; no fill except explicit popularity-fill ablation.',
        'graph':{
            'minimum_cooccurrence':[2,3,5], 'neighbors':100,
            'orientation':['candidate','input','union'],
            'neighbor_selection':['cooccurrence','lift'],'score_after_selection':'binary cosine',
            'unpruned_cosine':True,
            'asymmetric_cosine':{'candidate_support_exponent':[.25,.75],'locality_exponent':[1,2],'neighbors':100},
            'reviewer_activity_weight':['inverse_sqrt_degree','inverse_degree'],
            'rp3beta':{'alpha':1.,'beta':[.3,.6],'neighbors':100,
                'note':'Standard RP3beta is user-item-user-item recommendation. It reweights existing item cooccurrences; it cannot by itself connect item pairs with zero shared train users.'},
            'ease':{'regularization':[100,1000],'full_catalog_items':37610,'solver':'Exact positive-definite solve; candidate inverse columns may be solved in blocks; abort on resource/numerical failure without labeling a truncated catalog as full EASE.'},
            'ials':{'factors':[32,64],'regularization':.1,'confidence_scale':20,'iterations':10,'seed':20261108,'users':'Train positive catalog degree >=2','query':'Fixed-item-factor user fold-in from visible query positives only'},
            'metadata_to_collaborative_embedding':{'ridge_alpha':100,'target':'iALS item factors','blend_supported_prediction':[0.,.5,1.]},
            'positive_negative_relations':{'channels':['positive_positive','positive_negative'],'exclude_negative_negative':True,'negative_coefficient':[.1,.3]},
            'playtime_training_weight':'Per-game percentile rank of log1p snapshot playtime; weight .5+percentile; missing=.5; train accepted lineage only.',
            'time_training_weight':'Review-age exponential half-life 3 years relative to latest TRAIN review; weight in cooccurrence and support.'},
        'content':{
            'long_description':'Prefer C about_the_game, then detailed_description, then frozen A-first short description; content snapshot versions unchanged.',
            'missing_description_fill':'Only unsupported Linux items; pinned frank game metadata first, public Steam appdetails fallback; no review outcomes/statistics as content.',
            'word_tfidf':{'ngram_range':[1,2],'sublinear_tf':False,'min_df':1,'max_df':1.},
            'character_tfidf':{'analyzer':'char_wb','ngram_range':[3,5],'min_df':2,'sublinear_tf':True,'max_features':100000},
            'sentence_embedding':'Pinned multilingual model revision, actual inference, normalized item vectors and mean-profile cosine; record truncation and runtime.',
            'extended_metadata':['categories','developers','publishers'],'description_weight':.5,
            'train_review_body':'Accepted train accounts and review IDs only; deterministic cap 100 reviews per item; hash-based or source-order selection fixed before metrics, no validation/test review text.'},
        'query':{'aggregation':['mean_control','max_item_similarity','recency_half_life_1year','playtime_1_plus_log1p_minutes'],
            'recency_reference':'Latest visible query timestamp, never hidden target timestamp.',
            'negative_penalty':[.1,.3], 'negative_filter_cosine_threshold':[.5,.8],
            'negative_playtime_reliability':'Downweight negative input by 1/(1+log1p(minutes)/log1p(6000)); unavailable feature remains missing, no fabricated playtime.',
            'train_positive_rate_exponent':[.5,1.],
            'popularity_fill':'Only empty top10 slots, train positive counts, never replace personalized slots; report fill hits/users/slots.'},
        'fusion':{'method':'Reciprocal rank fusion','constant':60,'content_weight':[.25,.5,.75],
            'adaptive_content_weight':'weight/(1+candidate_train_support/1000); fixed threshold, not fit to frank.',
            'models':'Current v2-selected KNN and content; after validation selection, also compare the frozen best eligible new same-family models with the same grid.'},
        'reranking':{'model':'LambdaMART (LightGBM)','num_leaves':15,'n_estimators':[100,300],'learning_rate':.05,
            'features':['collaborative_score','content_score','train_log_support','train_positive_rate','release_year_distance','language_compatibility'],
            'candidate_generation':'Union of top50 from current KNN and content; never inject target into inference candidates.',
            'validation':'Deterministic user-key modulo 3 cross-fitting: each predicted user excluded from ranker fit. Unjudged candidates are weak relevance 0, not observed dislike.',
            'new_reference':'Refit fixed selected ranker on complete validation only after selection lock; frank remains reference.'},
        'auxiliary_evaluations':{
            'minimum_inputs':{'total':3,'positive':1,'seed':20261107},
            'multiple_targets':{'hidden':2,'remaining_total':5,'remaining_positive':1,'seed':20261107},
            'review_cap_validation':{'max_reviews_per_game':100,'seed':20261107,'label':'Reporting-only sample, not a reproduction of frank collection.'},
            'frank_input_expansion':['all_accepted_Kieran_history','accepted_Kieran_history_created_before_target'],
            'kieran_only_positive_exclusion_sensitivity':True,'reserved_account_Kieran_evaluation':True,
            'temporal_evaluation':'Use a global train cutoff and later held-out query-target chronology; if no independently train-fitted chronological model exists, label snapshot/sequence audit and do not claim future-prediction performance.',
            'duplicate_and_series_audit':'Exact normalized titles/descriptions and high-similarity pairs; do not automatically equate a series sequel with same game.',
            'usage':'Report current controls and frozen selected models, same-cohort paired comparisons only. Never use auxiliary/frank outcomes to choose primary model settings.'},
        'completion':'Every backlog row mapped to measured result, completed diagnosis, or concrete failed run/missing prerequisite with evidence; no invented scores, no silent skipped models.',
        'conditional_items':['ELSA only if full EASE fails from memory; SASRec feasibility after chronological validation; user_games coverage audit before any playtime augmentation'],
        'preserve':'Do not modify v1/v2 code, protocols, manifests, splits, models, or reports.'}


if __name__=='__main__':
    target=EDA/'data/reference/improvement_protocol_v3.json'
    if target.exists():raise FileExistsError('Do not rewrite a frozen protocol')
    write_json(target,protocol())
    print(digest(target))
