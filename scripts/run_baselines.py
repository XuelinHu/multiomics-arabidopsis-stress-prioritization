"""Homology-grouped development baselines; held-out test remains sealed."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

BASE=Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data')) / 'processed/protein_inputs'

def main():
    data=np.load(BASE/'inputs.npz')
    clusters=pd.read_csv(BASE/'homology_cluster.tsv',sep='\t',header=None,names=['cluster','protein'])
    assert not clusters.protein.duplicated().any()
    group=clusters.set_index('protein').loc[data['proteins'],'cluster'].to_numpy()
    split_path=BASE/'split.tsv'
    if not split_path.exists():
        raise FileNotFoundError('Fixed split.tsv is required. Obtain the released split or create it with a separately audited protocol; this script never rebuilds it.')
    split_frame=pd.read_csv(split_path,sep='\t')
    if not np.array_equal(split_frame.protein.to_numpy(),data['proteins']):
        raise ValueError('Existing split.tsv protein order does not match inputs.npz')
    split=split_frame.split.to_numpy()
    train=np.flatnonzero(split=='train'); val=np.flatnonzero(split=='validation')
    test=np.flatnonzero(split=='test')
    if (len(train),len(val),len(test)) != (16665,5203,5454):
        raise ValueError('Refusing non-final split sizes; expected 16665/5203/5454')
    if set(split_frame.loc[train,'cluster']) & set(split_frame.loc[val,'cluster']) or set(split_frame.loc[train,'cluster']) & set(split_frame.loc[test,'cluster']):
        raise ValueError('Existing split.tsv has cross-partition homology clusters')
    features={'composition':data['sequence_composition'],'control_expression':data['control_expression']}
    results=[]
    for seed in [17,42,2026]:
        for t,task in enumerate(['salt','mannitol_osmotic','heat']):
            y=data['labels'][:,t];mask=data['label_mask'][:,t]
            ti=train[mask[train]];vi=val[mask[val]]
            assert len(np.unique(y[ti]))==len(np.unique(y[vi]))==2
            for name,x in features.items():
                for algorithm in ['prior','logistic']:
                    model=DummyClassifier(strategy='prior') if algorithm=='prior' else make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed))
                    model.fit(x[ti],y[ti]); p=model.predict_proba(x[vi])[:,1]
                    results.append({'seed':seed,'task':task,'features':name,'model':algorithm,'split':'validation',
                        'n_train':len(ti),'n_validation':len(vi),'prevalence':float(y[vi].mean()),
                        'auprc':average_precision_score(y[vi],p),'auroc':roc_auc_score(y[vi],p),'f1_at_05':f1_score(y[vi],p>=.5)})
    out=Path('results/baselines');out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/'validation_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'split_sizes':{s:int((split==s).sum()) for s in np.unique(split)},
        'test_evaluated':False,'note':'Development baselines only. Deterministic logistic/prior seeds are reproducibility checks, not independent resampling. Homology threshold depends on recorded mmseqs invocation.'},indent=2))
    print(pd.DataFrame(results).groupby(['task','features','model']).auprc.mean().to_string())

if __name__=='__main__':main()
