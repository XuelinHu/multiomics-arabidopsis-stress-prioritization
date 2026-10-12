#!/usr/bin/env python3
"""Graph baselines: GCN, GAT, GraphSAGE, JK-Net, CrossRes, and CrossRes+JK.

The protocol matches the main experiment: concatenated ESM-PCA256, untreated
expression, and filtered domains; STRING>=700 training graph without validation
nodes; masked weighted BCE; validation macro-AUPRC early stopping; three seeds.
CrossRes follows KG-CrossResGNN with structural GCNConv and relational RGCNConv
branches and cross-branch residual exchange.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, tasks=3, layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, tasks)

    def forward(self, x, ei):
        x = self.proj(x)
        for c in self.convs:
            x = torch.nn.functional.gelu(c(x, ei))
        return self.head(x)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=128, tasks=3, heads=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.c1 = GATConv(hidden, hidden // heads, heads=heads)
        self.c2 = GATConv(hidden, hidden, heads=1)
        self.head = nn.Linear(hidden, tasks)

    def forward(self, x, ei):
        x = self.proj(x)
        x = torch.nn.functional.gelu(self.c1(x, ei))
        x = torch.nn.functional.gelu(self.c2(x, ei))
        return self.head(x)


class SAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, tasks=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.c1 = SAGEConv(hidden, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, tasks)

    def forward(self, x, ei):
        x = self.proj(x)
        x = torch.nn.functional.gelu(self.c1(x, ei))
        x = torch.nn.functional.gelu(self.c2(x, ei))
        return self.head(x)


class JKNet(nn.Module):
    """GCN with concatenated layer states via Jumping Knowledge."""
    def __init__(self, in_dim, hidden=128, tasks=3, layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden * (layers + 1), tasks)

    def forward(self, x, ei):
        x = self.proj(x)
        outs = [x]
        for c in self.convs:
            x = torch.nn.functional.gelu(c(x, ei))
            outs.append(x)
        return self.head(torch.cat(outs, dim=-1))


class CrossResidualLayer(nn.Module):
    """Structural GCN and relational RGCN with cross-branch residual exchange."""
    def __init__(self, hidden, num_relations=2):
        super().__init__()
        self.struct = GCNConv(hidden, hidden)
        self.rel = RGCNConv(hidden, hidden, num_relations)
        self.cross_s2r = nn.Linear(hidden, hidden)
        self.cross_r2s = nn.Linear(hidden, hidden)

    def forward(self, hs, hr, ei, et):
        hs_new = torch.nn.functional.gelu(self.struct(hs, ei))
        hr_new = torch.nn.functional.gelu(self.rel(hr, ei, et))
        hs = hs + hs_new + self.cross_r2s(hr_new)   # Relational to structural.
        hr = hr + hr_new + self.cross_s2r(hs_new)   # Structural to relational.
        return hs, hr


class CrossRes(nn.Module):
    def __init__(self, in_dim, hidden=128, tasks=3, layers=2, jk=False):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.s_in = nn.Linear(hidden, hidden)
        self.r_in = nn.Linear(hidden, hidden)
        self.layers = nn.ModuleList([CrossResidualLayer(hidden) for _ in range(layers)])
        self.jk = jk
        out_dim = hidden * 2 * (layers + 1) if jk else hidden * 2
        self.head = nn.Linear(out_dim, tasks)

    def forward(self, x, ei, et):
        x = self.proj(x)
        hs, hr = torch.relu(self.s_in(x)), torch.relu(self.r_in(x))
        outs = [torch.cat([hs, hr], dim=-1)]
        for layer in self.layers:
            hs, hr = layer(hs, hr, ei, et)
            outs.append(torch.cat([hs, hr], dim=-1))
        fused = torch.cat(outs, dim=-1) if self.jk else outs[-1]
        return self.head(fused)


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
        ei = torch.tensor([s, t], device=device)
        et = torch.tensor(ty, device=device)
        return ei, et
    tr_ei, tr_et = edges_for(ntrain)
    va_ei, va_et = edges_for(len(used))
    return x, (tr_ei, tr_et), (va_ei, va_et), ntrain, data, train, valid, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gcn', 'gat', 'sage', 'jk', 'cross', 'cross_jk'])
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--run-suffix', default='_v2')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = BASE/'processed/protein_inputs'
    x, (tr_ei, tr_et), (va_ei, va_et), ntrain, data, train, valid, used = build(directory, device, args.seed)
    if args.model == 'gcn':
        model = GCN(x.shape[1])
    elif args.model == 'gat':
        model = GAT(x.shape[1])
    elif args.model == 'sage':
        model = SAGE(x.shape[1])
    elif args.model == 'jk':
        model = JKNet(x.shape[1])
    elif args.model == 'cross':
        model = CrossRes(x.shape[1], jk=False)
    else:
        model = CrossRes(x.shape[1], jk=True)
    model = model.to(device)
    need_et = args.model in ('cross', 'cross_jk')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    run = Path('results/development')/f'{args.model}_seed{args.seed}{args.run_suffix}'
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
        logits = model(x[:ntrain], tr_ei, tr_et) if need_et else model(x[:ntrain], tr_ei)
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
            p = (model(x, va_ei, va_et) if need_et else model(x, va_ei))[ntrain:].sigmoid().cpu().numpy()
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
            print(args.model, args.seed, epoch, history[-1], flush=True)
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
        'seed': args.seed, 'model': args.model,
        'features': 'ESM-PCA256 + control expression + filtered domains',
        'ppi': 'STRING >=700', 'labels': 'v2', 'test_evaluated': False,
        'note': 'CrossRes follows the dual-branch cross-residual design of KG-CrossResGNN, adapted to node-level classification'}, indent=2))


if __name__ == '__main__':
    main()
