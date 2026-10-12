#!/usr/bin/env python3
"""Drought-label training extension using GSE220276 Bulk1 drought0 vs ww0 (3v3).

Rank labels use the top 25% log2FC as positives, bottom 25% as controls,
and mask the middle. Drought is the fourth ModalityGraph task (domf setup).
GSE220276 does not contribute to scaling, PCA, or vocabulary fitting.
"""
import argparse
import copy
import gzip
import json
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.modality_graph import ModalityGraph

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
RESULTS = Path('results')


def drought_labels(proteins_df):
    cols = {}
    with gzip.open(f"{BASE}/raw/geo/GSE220276/GSE220276_Bulk1_Raw_Counts.txt.gz", "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, c in enumerate(header[1:], start=1):
            cols[c] = i
    d30 = ["Bulk1_D30-1", "Bulk1_D30-2", "Bulk1_D30-3"]
    ww0 = ["Bulk1_ww-0-1", "Bulk1_ww-0-2", "Bulk1_ww-0-3"]
    fc = {}
    with gzip.open(f"{BASE}/raw/geo/GSE220276/GSE220276_Bulk1_Raw_Counts.txt.gz", "rt") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            g = parts[0]
            if not g.startswith("AT"):
                continue
            d = np.array([float(parts[cols[c]]) for c in d30])
            w = np.array([float(parts[cols[c]]) for c in ww0])
            fc[g] = math.log2((d.mean() + 1) / (w.mean() + 1))
    vals = np.array([fc.get(g, np.nan) for g in proteins_df.gene])
    finite = np.isfinite(vals)
    pos_thr = np.quantile(vals[finite], 0.75)
    neg_thr = np.quantile(vals[finite], 0.25)
    labels = ((vals >= pos_thr)).astype(float)
    masks = (vals >= pos_thr) | (vals <= neg_thr)
    labels[~masks] = 0.0
    print(f"Drought labels: {int(labels.sum())} positives, {int(masks.sum())-int(labels.sum())} controls, "
          f"{int((~masks).sum())} masked (top25%>={pos_thr:.2f}, bottom25%<={neg_thr:.2f})")
    return labels.astype(float), masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--run-suffix', default='_v2_d')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = BASE/'processed/protein_inputs'
    data = np.load(directory/'inputs.npz')
    proteins = pd.read_csv(directory/'proteins.tsv', sep='\t').fillna('')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    train = np.flatnonzero(split.split == 'train')
    valid = np.flatnonzero(split.split == 'validation')
    used = np.concatenate([train, valid])
    ntrain = len(train)
    dlab, dmask = drought_labels(proteins)
    labels = np.concatenate([data['labels'], dlab[:, None]], axis=1)
    masks = np.concatenate([data['label_mask'], dmask[:, None]], axis=1)
    # Fit features on training proteins as in the main experiment.
    esm = np.load(directory/'esm_embeddings.npz')
    esm_x = esm['embeddings'].astype('float32')
    s1 = StandardScaler().fit(esm_x[train])
    pca = PCA(n_components=256, random_state=args.seed).fit(s1.transform(esm_x[train]))
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
    tensors = [torch.tensor(seq, device=device), torch.tensor(expr, device=device),
               torch.tensor(dom[used], device=device)]
    lookup = {f'3702.{p}': i for i, p in enumerate(data['proteins'][used])}
    rr, cc, ww = [], [], []
    for chunk in pd.read_csv(BASE/'raw/string/3702.protein.links.v12.0.txt.gz', sep=' ', chunksize=200000):
        chunk = chunk[chunk.combined_score >= 700]
        for a, b, w in chunk.itertuples(index=False, name=None):
            i, j = lookup.get(a), lookup.get(b)
            if i is None or j is None or i == j:
                continue
            rr.append(i)
            cc.append(j)
            ww.append(w/1000)
    adjacency = sparse.csr_matrix((ww, (rr, cc)), shape=(len(used), len(used)), dtype='float32')
    adjacency = adjacency.maximum(adjacency.T)

    def normalize(a):
        a = a + sparse.eye(a.shape[0], dtype='float32')
        a = sparse.diags(1/np.asarray(a.sum(1)).ravel()) @ a
        a = a.tocoo()
        return torch.sparse_coo_tensor(np.stack([a.row, a.col]), a.data, a.shape, device=device).coalesce()
    train_graph = normalize(adjacency[:ntrain, :ntrain])
    val_graph = normalize(adjacency)
    label_t = torch.tensor(labels[used], device=device)
    mask_t = torch.tensor(masks[used], device=device)
    positive = (label_t[:ntrain] * mask_t[:ntrain]).sum(0)
    negative = ((1 - label_t[:ntrain]) * mask_t[:ntrain]).sum(0)
    assert (positive > 0).all() and (negative > 0).all()
    model = ModalityGraph([x.shape[1] for x in tensors], use_graph=True, tasks=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    run = RESULTS/'development'/f'modality_graph_seed{args.seed}{args.run_suffix}'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'metrics.json').exists():
        raise RuntimeError('Completed run exists; preserve it')
    best = -1
    stall = 0
    history = []
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model([x[:ntrain] for x in tensors], train_graph)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, label_t[:ntrain], pos_weight=negative/positive, reduction='none')
        loss = (losses * mask_t[:ntrain]).sum()/mask_t[:ntrain].sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            p = model(tensors, val_graph)[0][ntrain:].sigmoid().cpu().numpy()
        yt = labels[valid]
        mt = masks[valid]
        ap = [average_precision_score(yt[mt[:, t], t], p[mt[:, t], t]) for t in range(4)]
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
    for t, task in enumerate(['salt', 'mannitol_osmotic', 'heat', 'drought']):
        y = yt[mt[:, t], t]
        p = best_prediction[mt[:, t], t]
        metrics[task] = {'auprc': average_precision_score(y, p), 'auroc': roc_auc_score(y, p),
                         'n': len(y), 'prevalence': float(y.mean())}
    (run/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (run/'config.json').write_text(json.dumps({
        'seed': args.seed, 'tasks': 4,
        'drought_label': 'GSE220276 Bulk1 drought0 vs ww0 log2FC rank-based: top25% positive, bottom25% comparator',
        'note': '3v3 contrast, rank-based labels are exploratory; GSE220276 not used for feature fitting',
        'test_evaluated': False}, indent=2))


if __name__ == '__main__':
    main()
