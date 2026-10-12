"""R-GCN graph baseline: two RGCNConv layers (forward/reverse PPI) on STRING>=700.

Features match the main model: ESM-PCA256, control expression, and filtered domains.
The protocol matches train_modality_graph.py: training graph excludes validation nodes,
early stopping uses validation data, and three seeds are evaluated.
"""
import argparse
import copy
import json
import random
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch_geometric.nn import RGCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden=128, tasks=3, num_relations=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv1 = RGCNConv(hidden, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.head = nn.Linear(hidden, tasks)

    def forward(self, x, edge_index, edge_type):
        x = self.proj(x)
        x = torch.nn.functional.gelu(self.conv1(x, edge_index, edge_type))
        x = torch.nn.functional.gelu(self.conv2(x, edge_index, edge_type))
        return self.head(x)


def build(directory, device, seed):
    data = np.load(directory/'inputs.npz')
    proteins = pd.read_csv(directory/'proteins.tsv', sep='\t').fillna('')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    assert np.array_equal(data['proteins'], split.protein) and np.array_equal(data['proteins'], proteins.protein)
    train = np.flatnonzero(split.split == 'train')
    valid = np.flatnonzero(split.split == 'validation')
    used = np.concatenate([train, valid])
    ntrain = len(train)
    lookup = {f'3702.{p}': i for i, p in enumerate(data['proteins'][used])}
    # Fit feature transformations on training proteins.
    esm = np.load(directory/'esm_embeddings.npz')
    esm_x = esm['embeddings'].astype('float32')
    s1 = StandardScaler().fit(esm_x[train])
    pca = PCA(n_components=256, random_state=seed).fit(s1.transform(esm_x[train]))
    seq = pca.transform(s1.transform(esm_x[used])).astype('float32')
    s2 = StandardScaler().fit(data['control_expression'][train])
    expr = s2.transform(data['control_expression'][used]).astype('float32')
    domains = [set(filter(None, s.split(';'))) for s in proteins.InterPro]
    from collections import Counter
    freq = Counter()
    for i in train:
        freq.update(domains[i])
    keep = [c for c in sorted(freq) if freq[c] >= 20]
    idx = {c: i for i, c in enumerate(keep)}
    rows = np.zeros((len(domains), len(keep)), dtype='float32')
    counts = np.zeros((len(domains), 1), dtype='float32')
    flags = np.zeros((len(domains), 1), dtype='float32')
    for i, d in enumerate(domains):
        for c in d & set(keep):
            rows[i, idx[c]] = 1.0
        counts[i, 0] = np.log1p(len(d & set(keep)))
        flags[i, 0] = 1.0 if d else 0.0
    dom = np.concatenate([rows, counts, flags], axis=1).astype('float32')
    x = torch.tensor(np.concatenate([seq, expr, dom[used]], axis=1), device=device)
    # Two edge relations: forward and reverse.
    rr, cc = [], []
    for chunk in pd.read_csv(BASE/'raw/string/3702.protein.links.v12.0.txt.gz', sep=' ', chunksize=200000):
        chunk = chunk[chunk.combined_score >= 700]
        for a, b, w in chunk.itertuples(index=False, name=None):
            i, j = lookup.get(a), lookup.get(b)
            if i is None or j is None or i == j:
                continue
            rr.append(i)
            cc.append(j)
    def edges_for(n):
        s, t, ty = [], [], []
        for a, b in zip(rr, cc):
            if a < n and b < n:
                s += [a, b]
                t += [b, a]
                ty += [0, 1]
        return torch.tensor([s, t], device=device), torch.tensor(ty, device=device)
    train_ei, train_et = edges_for(ntrain)
    val_ei, val_et = edges_for(len(used))
    return x, train_ei, train_et, val_ei, val_et, ntrain, data, train, valid, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--run-suffix', default='_v2')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = BASE/'processed/protein_inputs'
    x, train_ei, train_et, val_ei, val_et, ntrain, data, train, valid, used = build(directory, device, args.seed)
    model = RGCN(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    run = Path('results/development')/f'rgcn_seed{args.seed}{args.run_suffix}'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'metrics.json').exists():
        raise RuntimeError('Completed run exists; preserve it')
    labels = torch.tensor(data['labels'][used], device=device)
    masks = torch.tensor(data['label_mask'][used], device=device)
    positive = (labels[:ntrain] * masks[:ntrain]).sum(0)
    negative = ((1 - labels[:ntrain]) * masks[:ntrain]).sum(0)
    best = -1
    stall = 0
    history = []
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x[:ntrain], train_ei, train_et)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[:ntrain], pos_weight=negative/positive, reduction='none')
        loss = (losses * masks[:ntrain]).sum()/masks[:ntrain].sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            p = model(x, val_ei, val_et)[ntrain:].sigmoid().cpu().numpy()
        yt = data['labels'][valid]
        mt = data['label_mask'][valid]
        ap = [average_precision_score(yt[mt[:, t], t], p[mt[:, t], t]) for t in range(3)]
        score = float(np.mean(ap))
        history.append({'epoch': epoch, 'loss': float(loss.detach()), 'validation_macro_auprc': score})
        pd.DataFrame(history).to_csv(run/'history.csv', index=False)
        if score > best:
            best = score
            stall = 0
            best_state = copy.deepcopy(model.state_dict())
            best_prediction = p.copy()
            torch.save(best_state, run/'best.pt')
        else:
            stall += 1
        if epoch % 10 == 0:
            print(args.seed, epoch, history[-1], flush=True)
        if stall >= 20:
            break
    metrics = {}
    for t, task in enumerate(['salt', 'mannitol_osmotic', 'heat']):
        y = yt[mt[:, t], t]
        p = best_prediction[mt[:, t], t]
        metrics[task] = {'auprc': average_precision_score(y, p), 'auroc': roc_auc_score(y, p),
                         'n': len(y), 'prevalence': float(y.mean())}
    (run/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (run/'config.json').write_text(json.dumps({
        'seed': args.seed, 'model': 'RGCN 2-layer, 2 relations (forward+reverse PPI)',
        'features': 'ESM-PCA256 + control expression + filtered domains', 'test_evaluated': False,
        'ppi': 'STRING >=700', 'labels': 'v2'}, indent=2))
    np.savez_compressed(run/'validation_predictions.npz', proteins=data['proteins'][valid],
                        predictions=best_prediction, labels=yt, mask=mt)


if __name__ == '__main__':
    main()
