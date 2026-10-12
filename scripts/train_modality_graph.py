"""Train development runs with fixed splits; never evaluate the test partition."""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.modality_graph import ModalityGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', PROJECT_ROOT/'data'))


def parse_thresholds(value):
    """Parse S=Q:B,M=Q:B,H=Q:B without changing the frozen input archive."""
    thresholds = {'S': (0.05, 0.0), 'M': (0.05, 0.5), 'H': (0.001, 1.5)}
    if not value:
        return thresholds
    for item in value.split(','):
        task, values = item.split('=', 1)
        q_value, effect = values.split(':', 1)
        task = task.strip().upper()
        if task not in thresholds:
            raise argparse.ArgumentTypeError(f'Unknown task in thresholds: {task}')
        thresholds[task] = (float(q_value), float(effect))
    return thresholds


def labels_for_thresholds(directory, proteins, thresholds):
    differential = pd.read_csv(
        BASE/'processed/GSE146206/author_differential_results.tsv.gz', sep='\t'
    ).set_index('AGI').loc[proteins.gene]
    labels = np.zeros((len(proteins), 3), dtype='float32')
    masks = np.zeros((len(proteins), 3), dtype=bool)
    for column, task in enumerate('SMH'):
        q_value = differential[f'qvalue_{task}'].to_numpy(float)
        effect = differential[f'b_{task}'].to_numpy(float)
        q_cutoff, effect_cutoff = thresholds[task]
        positive = (np.isfinite(q_value) & np.isfinite(effect) &
                    (q_value <= q_cutoff) & (np.abs(effect) >= effect_cutoff))
        comparator = np.isfinite(q_value) & (q_value >= 0.5)
        labels[:, column] = positive
        masks[:, column] = positive | comparator
    return labels, masks


def select_training_subset(train, split, fraction, subset_seed):
    if fraction >= 1:
        return train
    if not 0 < fraction < 1:
        raise ValueError('--train-fraction must be in (0, 1]')
    clusters = split.cluster.to_numpy()[train]
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(subset_seed)
    rng.shuffle(unique_clusters)
    selected = set(unique_clusters[:max(1, int(np.ceil(len(unique_clusters)*fraction)))])
    subset = train[np.array([cluster in selected for cluster in clusters])]
    if not len(subset):
        raise RuntimeError('Cluster-aware training subset is empty')
    return subset


def load_ppi_edges(data, node_indices, mode):
    lookup = {f'3702.{protein}': i for i, protein in enumerate(data['proteins'][node_indices])}
    rows, columns, weights = [], [], []
    if mode == 'none':
        return rows, columns, weights
    if mode == 'combined':
        path = BASE/'raw/string/3702.protein.links.v12.0.txt.gz'
        chunks = pd.read_csv(path, sep=r'\s+', chunksize=200000)
        for chunk in chunks:
            chunk = chunk[chunk.combined_score >= 700]
            for a, b, score in chunk[['protein1', 'protein2', 'combined_score']].itertuples(index=False, name=None):
                i, j = lookup.get(a), lookup.get(b)
                if i is not None and j is not None and i != j:
                    rows.append(i); columns.append(j); weights.append(score/1000)
        return rows, columns, weights

    path = BASE/'raw/string/3702.protein.links.detailed.v12.0.txt.gz'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} is required for --ppi-mode {mode}. Download the STRING detailed-links file.'
        )
    evidence_columns = {
        'experiments': ['experimental'],
        'experiments_coexpression': ['experimental', 'coexpression'],
    }[mode]
    for chunk in pd.read_csv(path, sep=r'\s+', chunksize=200000):
        evidence = chunk[evidence_columns].max(axis=1)
        chunk = chunk.assign(_evidence=evidence)
        chunk = chunk[chunk._evidence >= 700]
        for a, b, score in chunk[['protein1', 'protein2', '_evidence']].itertuples(index=False, name=None):
            i, j = lookup.get(a), lookup.get(b)
            if i is not None and j is not None and i != j:
                rows.append(i); columns.append(j); weights.append(score/1000)
    return rows, columns, weights

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--sequence-mode',choices=['composition','esm'],default='composition',
                        help='esm: ESM-2 t33_650M 嵌入经 PCA(256, 仅训练集拟合) 替换序列组成')
    parser.add_argument('--ablate',choices=['none','sequence','expression','domain','go'],default='none',
                        help='去掉一个特征模态（消融）；PPI 消融用 --no-graph')
    parser.add_argument('--run-suffix',default='',help='附加到 run 目录名（区分标签版本等）')
    parser.add_argument('--domain-mode',choices=['multihot','filtered'],default='multihot',
                        help='filtered: 仅训练集频率>=20 的 IPR 类 + 域数覆盖特征 + 未注释指示位')
    parser.add_argument('--go-mode',choices=['none','filtered'],default='none',
                        help='filtered: GO 词表(训练集频率>=50) multi-hot + log GO 数 + 未注释指示位，作为第 4 模态')
    parser.add_argument('--data-root', type=Path, help='Override KGMULTIOMICS_DATA_ROOT')
    parser.add_argument('--output-root', type=Path, default=Path('results/development'))
    parser.add_argument('--thresholds', default='',
                        help='In-memory labels, e.g. S=0.05:0,M=0.05:0.5,H=0.001:1.5')
    parser.add_argument('--train-fraction', type=float, default=1.0,
                        help='Cluster-aware fraction of training clusters; validation remains fixed')
    parser.add_argument('--subset-seed', type=int, default=20261004)
    parser.add_argument('--shuffle-labels', action='store_true',
                        help='Permute observed training labels within each task; validation labels stay real')
    parser.add_argument('--ppi-mode', choices=['combined','experiments','experiments_coexpression','none'],
                        default='combined')
    args=parser.parse_args()
    global BASE
    if args.data_root:
        BASE = args.data_root.resolve()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    directory=BASE/'processed/protein_inputs'
    data=np.load(directory/'inputs.npz')
    proteins=pd.read_csv(directory/'proteins.tsv',sep='\t').fillna('')
    split=pd.read_csv(directory/'split.tsv',sep='\t')
    assert np.array_equal(data['proteins'],split.protein) and np.array_equal(data['proteins'],proteins.protein)
    full_train=np.flatnonzero(split.split=='train');valid=np.flatnonzero(split.split=='validation')
    train=select_training_subset(full_train, split, args.train_fraction, args.subset_seed)
    # Fully inductive development: held-out test nodes and edges are absent.
    used=np.concatenate([train,valid]);ntrain=len(train)
    domains=[set(filter(None,s.split(';'))) for s in proteins.InterPro]
    mlb=MultiLabelBinarizer();mlb.fit([domains[i] for i in train])
    known=set(mlb.classes_)
    if args.domain_mode=='multihot':
        domain_matrix=mlb.transform([d&known for d in domains]).astype('float32')
    else:
        # filtered: 仅训练集频率>=20 的 IPR 类 + 注释域数(log) + 是否未注释指示位
        from collections import Counter
        freq=Counter()
        for i in train: freq.update(domains[i])
        keep=[c for c in mlb.classes_ if freq[c]>=20]
        idx={c:i for i,c in enumerate(keep)}
        rows=np.zeros((len(domains),len(keep)),dtype='float32')
        counts=np.zeros((len(domains),1),dtype='float32')
        flags=np.zeros((len(domains),1),dtype='float32')
        for i,d in enumerate(domains):
            for c in d&set(keep): rows[i,idx[c]]=1.0
            counts[i,0]=np.log1p(len(d&known))
            flags[i,0]=1.0 if d else 0.0
        domain_matrix=np.concatenate([rows,counts,flags],axis=1).astype('float32')
    assert domain_matrix.shape[1]>0
    mods={};scalers=[];sequence_feature_desc='composition baseline'
    if args.sequence_mode=='composition':
        scaler=StandardScaler().fit(data['sequence_composition'][train])
        scalers.append({'feature':'sequence_composition','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()})
        mods['sequence']=torch.tensor(scaler.transform(data['sequence_composition'][used]),dtype=torch.float32,device=device)
    else:
        esm=np.load(directory/'esm_embeddings.npz')
        assert np.array_equal(esm['proteins'],data['proteins']),'ESM ID 顺序与 inputs.npz 不一致'
        esm_x=esm['embeddings'].astype('float32')
        # 泄漏规则：标准化参数与 PCA 仅用训练集拟合
        esm_scaler=StandardScaler().fit(esm_x[train])
        pca=PCA(n_components=256,random_state=args.seed).fit(esm_scaler.transform(esm_x[train]))
        esm_proj=pca.transform(esm_scaler.transform(esm_x[used])).astype('float32')
        scalers.append({'feature':'esm_pca256','mean':esm_scaler.mean_.tolist(),'scale':esm_scaler.scale_.tolist(),
                        'pca_explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum())})
        sequence_feature_desc=(f'ESM-2 t33_650M mean-pooled 1280d -> PCA 256d (fit on train only), '
                               f'cumulative explained variance {pca.explained_variance_ratio_.sum():.3f}')
        mods['sequence']=torch.tensor(esm_proj,device=device)
    scaler=StandardScaler().fit(data['control_expression'][train])
    scalers.append({'feature':'control_expression','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()})
    mods['expression']=torch.tensor(scaler.transform(data['control_expression'][used]),dtype=torch.float32,device=device)
    mods['domain']=torch.tensor(domain_matrix[used],device=device)
    if args.go_mode=='filtered':
        go_terms={}
        for line in open(BASE/'raw/go/uniprot_arabidopsis_go.tsv'):
            if line.startswith('Entry'):continue
            e,_,gos=line.rstrip('\n').partition('\t')
            go_terms[e]=set(filter(None,gos.split('; ')))
        from collections import Counter as _Counter
        gf=_Counter()
        for i in train: gf.update(go_terms.get(data['proteins'][i],set()))
        gov=[t for t,c in gf.items() if c>=50]
        gi={t:i for i,t in enumerate(gov)}
        go_rows=np.zeros((len(data['proteins']),len(gov)),dtype='float32')
        go_cnt=np.zeros((len(data['proteins']),1),dtype='float32')
        go_flag=np.zeros((len(data['proteins']),1),dtype='float32')
        for i,p in enumerate(data['proteins']):
            ts=go_terms.get(p,set())&set(gov)
            for t in ts: go_rows[i,gi[t]]=1.0
            go_cnt[i,0]=np.log1p(len(ts))
            go_flag[i,0]=1.0 if go_terms.get(p,set()) else 0.0
        go_matrix=np.concatenate([go_rows,go_cnt,go_flag],axis=1).astype('float32')
        mods['go']=torch.tensor(go_matrix[used],device=device)
        scalers.append({'feature':'go_filtered','go_vocab_size':len(gov)})
    if args.ablate!='none':
        assert args.ablate in mods, f'无法消融 {args.ablate}'
        del mods[args.ablate]
    tensors=list(mods.values())
    ppi_mode = 'none' if args.no_graph else args.ppi_mode
    rr,cc,ww=load_ppi_edges(data, used, ppi_mode)
    adjacency=sparse.csr_matrix((ww,(rr,cc)),shape=(len(used),len(used)),dtype='float32')
    adjacency=adjacency.maximum(adjacency.T)
    # Training never aggregates validation nodes. Validation sees development graph only.
    def normalize(a):
        a=a+sparse.eye(a.shape[0],dtype='float32')
        a=sparse.diags(1/np.asarray(a.sum(1)).ravel())@a
        a=a.tocoo()
        return torch.sparse_coo_tensor(np.stack([a.row,a.col]),a.data,a.shape,device=device).coalesce()
    train_graph=normalize(adjacency[:ntrain,:ntrain]);val_graph=normalize(adjacency)
    thresholds=parse_thresholds(args.thresholds)
    labels_np,masks_np=labels_for_thresholds(directory, proteins, thresholds)
    labels=torch.tensor(labels_np[used],device=device)
    masks=torch.tensor(masks_np[used],device=device)
    if args.shuffle_labels:
        rng=np.random.default_rng(args.seed)
        shuffled=labels[:ntrain].clone()
        for task in range(shuffled.shape[1]):
            observed=torch.nonzero(masks[:ntrain,task], as_tuple=True)[0].cpu().numpy()
            values=shuffled[observed,task].cpu().numpy()
            rng.shuffle(values)
            shuffled[observed,task]=torch.tensor(values,device=device)
        labels[:ntrain]=shuffled
    positive=(labels[:ntrain]*masks[:ntrain]).sum(0)
    negative=((1-labels[:ntrain])*masks[:ntrain]).sum(0)
    if not ((positive>0).all() and (negative>0).all()):
        if not args.thresholds:
            raise ValueError('Primary training labels must contain both classes in every task')
        run=args.output_root/f'modality_graph_seed{args.seed}_esm{args.run_suffix}'
        run.mkdir(parents=True,exist_ok=True)
        if (run/'metrics.json').exists():
            raise RuntimeError('Completed run exists; preserve it')
        tasks=['salt','mannitol_osmotic','heat']
        unavailable={}
        for t,task in enumerate(tasks):
            y=labels_np[valid,t][masks_np[valid,t]]
            unavailable[task]={'auprc':None,'auroc':None,'n':len(y),'prevalence':float(y.mean())}
        (run/'metrics.json').write_text(json.dumps(unavailable,indent=2))
        (run/'config.json').write_text(json.dumps({
            'seed':args.seed,'test_evaluated':False,'status':'infeasible_labels',
            'reason':'At least one training task has no positive or no comparator labels; fixed three-task experiment not fitted.',
            'positive_training_labels':positive.cpu().tolist(),'comparator_training_labels':negative.cpu().tolist(),
            'label_thresholds':{task:{'q':values[0],'effect':values[1]} for task,values in thresholds.items()},
            'train_fraction':args.train_fraction,'shuffle_labels':False,'ppi_mode':ppi_mode,
            'edges':int(adjacency.nnz),'training_edges':int(adjacency[:ntrain,:ntrain].nnz),
            'graph_node_coverage':float(np.mean(np.asarray(adjacency.sum(1)).ravel()>0)),
            'training_graph_node_coverage':float(np.mean(np.asarray(adjacency[:ntrain,:ntrain].sum(1)).ravel()>0)),
        },indent=2))
        pd.DataFrame(columns=['epoch','loss','validation_macro_auprc']).to_csv(run/'history.csv',index=False)
        print(f'{run}: infeasible fixed labels, training skipped',flush=True)
        return
    model=ModalityGraph([x.shape[1] for x in tensors],use_graph=ppi_mode!='none').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    suffix=('_esm' if args.sequence_mode=='esm' else '')+('' if args.ablate=='none' else f'_no{args.ablate}')+args.run_suffix
    run=args.output_root/f'{"gated_mlp" if ppi_mode=="none" else "modality_graph"}_seed{args.seed}{suffix}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.json').exists():raise RuntimeError('Completed run exists; preserve it')
    best=-1;stall=0;history=[];best_state=None
    for epoch in range(args.epochs):
        model.train();optimizer.zero_grad()
        logits,_=model([x[:ntrain] for x in tensors],train_graph)
        losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[:ntrain],pos_weight=negative/positive,reduction='none')
        loss=(losses*masks[:ntrain]).sum()/masks[:ntrain].sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():p=model(tensors,val_graph)[0][ntrain:].sigmoid().cpu().numpy()
        yt=labels_np[valid];mt=masks_np[valid]
        ap=[average_precision_score(yt[mt[:,t],t],p[mt[:,t],t]) for t in range(3)]
        score=float(np.mean(ap))
        history.append({'epoch':epoch,'loss':float(loss.detach()),'validation_macro_auprc':score})
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        if score>best:
            best=score;stall=0;best_state=copy.deepcopy(model.state_dict());best_prediction=p.copy()
            torch.save(best_state,run/'best.pt')
        else:stall+=1
        if epoch%10==0:print(args.seed,epoch,history[-1],flush=True)
        if stall>=20:break
    metrics={}
    for t,task in enumerate(['salt','mannitol_osmotic','heat']):
        y=yt[mt[:,t],t];p=best_prediction[mt[:,t],t]
        metrics[task]={'auprc':average_precision_score(y,p),'auroc':roc_auc_score(y,p),'n':len(y),'prevalence':float(y.mean())}
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (run/'config.json').write_text(json.dumps({'seed':args.seed,'epochs_max':args.epochs,'domain_vocabulary':list(mlb.classes_),
        'scalers':scalers,'test_evaluated':False,'status':'completed',
        'positive_training_labels':positive.cpu().tolist(),'comparator_training_labels':negative.cpu().tolist(),
        'sequence_features':sequence_feature_desc,'ablated':args.ablate,
        'domain_mode':args.domain_mode,'domain_dim':int(domain_matrix.shape[1]),
        'modalities':list(mods.keys()),'ppi_mode':ppi_mode,'ppi_threshold':700,
        'label_thresholds':{task:{'q':values[0],'effect':values[1]} for task,values in thresholds.items()},
        'train_fraction':args.train_fraction,'training_nodes':int(ntrain),
        'subset_seed':args.subset_seed,'shuffle_labels':args.shuffle_labels,
        'training_graph':'train nodes only','validation_graph':'train+validation nodes',
        'edges':int(adjacency.nnz), 'training_edges':int(adjacency[:ntrain,:ntrain].nnz),
        'graph_node_coverage':float(np.mean(np.asarray(adjacency.sum(axis=1)).ravel()>0)),
        'training_graph_node_coverage':float(np.mean(np.asarray(adjacency[:ntrain,:ntrain].sum(axis=1)).ravel()>0))},indent=2))
    np.savez_compressed(run/'validation_predictions.npz',proteins=data['proteins'][valid],predictions=best_prediction,labels=yt,mask=mt)

if __name__=='__main__':main()
