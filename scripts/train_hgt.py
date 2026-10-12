"""Train HGT on the development graph; never evaluate the test partition.

Same split/early-stopping/metrics protocol as train_modality_graph.py.
Protein features: ESM-2 PCA256 (train-fit) + control expression (standardized).
Graph: PPI>=700, protein->InterPro domain, protein->GO term (vocab from train only).
Training graph = train proteins only; validation graph = train+validation proteins.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.hgt import HGT

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))


def build_graphs(directory, device, seed):
    data = np.load(directory/'inputs.npz')
    proteins = pd.read_csv(directory/'proteins.tsv', sep='\t').fillna('')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    assert np.array_equal(data['proteins'], split.protein) and np.array_equal(data['proteins'], proteins.protein)
    train = np.flatnonzero(split.split == 'train')
    valid = np.flatnonzero(split.split == 'validation')
    used = np.concatenate([train, valid])
    ntrain = len(train)
    lookup = {f'3702.{p}': i for i, p in enumerate(data['proteins'][used])}

    # --- protein features: ESM-PCA256 + control expression (train-fit) ---
    esm = np.load(directory/'esm_embeddings.npz')
    assert np.array_equal(esm['proteins'], data['proteins'])
    esm_x = esm['embeddings'].astype('float32')
    s1 = StandardScaler().fit(esm_x[train])
    pca = PCA(n_components=256, random_state=seed).fit(s1.transform(esm_x[train]))
    seq = pca.transform(s1.transform(esm_x[used])).astype('float32')
    s2 = StandardScaler().fit(data['control_expression'][train])
    expr = s2.transform(data['control_expression'][used]).astype('float32')
    x_protein = torch.tensor(np.concatenate([seq, expr], axis=1), device=device)

    # --- domain nodes: train InterPro vocabulary ---
    domains = [set(filter(None, s.split(';'))) for s in proteins.InterPro]
    domain_vocab = sorted({d for i in train for d in domains[i]})
    domain_idx = {d: i for i, d in enumerate(domain_vocab)}
    # --- GO nodes: terms with train frequency >= 50 ---
    go_terms = {}
    for line in open(BASE/'raw/go/uniprot_arabidopsis_go.tsv'):
        if line.startswith('Entry'):
            continue
        e, _, gos = line.rstrip('\n').partition('\t')
        go_terms[e] = set(filter(None, gos.split('; ')))
    from collections import Counter
    go_freq = Counter()
    for i in train:
        go_freq.update(go_terms.get(data['proteins'][i], set()))
    go_vocab = sorted({t for t, c in go_freq.items() if c >= 50})
    go_idx = {t: i for i, t in enumerate(go_vocab)}
    n_domain, n_go = len(domain_vocab), len(go_vocab)

    # --- edges ---
    def add(srcs, dsts, dtype):
        edges[dtype].append((srcs, dsts))
    edges = {t: [] for t in [('protein', 'ppi', 'protein'), ('protein', 'contains', 'domain'),
                             ('protein', 'associated', 'go')]}
    for i in range(len(used)):
        p = data['proteins'][used][i]
        for d in domains[used[i]] & set(domain_vocab):
            add(i, domain_idx[d], ('protein', 'contains', 'domain'))
        for g in go_terms.get(p, set()) & set(go_vocab):
            add(i, go_idx[g], ('protein', 'associated', 'go'))
    for chunk in pd.read_csv(BASE/'raw/string/3702.protein.links.v12.0.txt.gz', sep=' ', chunksize=200000):
        chunk = chunk[chunk.combined_score >= 700]
        for a, b, w in chunk.itertuples(index=False, name=None):
            i, j = lookup.get(a), lookup.get(b)
            if i is None or j is None or i == j:
                continue
            add(i, j, ('protein', 'ppi', 'protein'))
            add(j, i, ('protein', 'ppi', 'protein'))

    def to_dict(mask_fn):
        out = {}
        for dtype, pairs in edges.items():
            s, t = zip(*pairs) if pairs else ((), ())
            s = np.array(s, dtype=np.int64)
            t = np.array(t, dtype=np.int64)
            keep = mask_fn(dtype, s, t)
            out[dtype] = torch.stack([torch.tensor(s[keep]), torch.tensor(t[keep])], dim=0).to(device)
        return out

    def mask_train(dtype, s, t):
        if dtype == ('protein', 'ppi', 'protein'):
            return (s < ntrain) & (t < ntrain)
        return s < ntrain  # Domain/GO edge source must be a training protein.

    def mask_all(dtype, s, t):
        return np.ones(len(s), dtype=bool)

    return (x_protein, edges, to_dict(mask_train), to_dict(mask_all), ntrain,
            domain_vocab, go_vocab, n_domain, n_go, data, train, valid, used)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--run-suffix', default='_v2')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = BASE/'processed/protein_inputs'
    (x_protein, edges, train_graph, val_graph, ntrain, domain_vocab, go_vocab,
     n_domain, n_go, data, train, valid, used) = build_graphs(directory, device, args.seed)

    model = HGT(x_protein.shape[1], hidden=args.hidden, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    run = Path("results/development")/f"hgt_seed{args.seed}{args.run_suffix}"
    run.mkdir(parents=True, exist_ok=True)
    if (run/'metrics.json').exists():
        raise RuntimeError('Completed run exists; preserve it')
    labels = torch.tensor(data['labels'][used], device=device)
    masks = torch.tensor(data['label_mask'][used], device=device)
    positive = (labels[:ntrain] * masks[:ntrain]).sum(0)
    negative = ((1 - labels[:ntrain]) * masks[:ntrain]).sum(0)
    assert (positive > 0).all() and (negative > 0).all(), 'Degenerate training labels'
    x_dict = {'protein': x_protein}
    best = -1
    stall = 0
    history = []
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model({k: v[:ntrain] for k, v in x_dict.items()}, train_graph, n_domain, n_go)
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
            p = model(x_dict, val_graph, n_domain, n_go)[ntrain:].sigmoid().cpu().numpy()
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
        'seed': args.seed, 'epochs_max': args.epochs, "hidden": args.hidden, "heads": args.heads,
        'protein_features': 'ESM-2 t33_650M PCA256 (train-fit) + control expression standardized',
        'n_domain_nodes': n_domain, 'n_go_nodes': n_go,
        'domain_vocab': 'InterPro classes observed in train proteins',
        'go_vocab': 'GO terms with train frequency >= 50',
        'ppi': 'STRING combined score >=700, unweighted',
        'labels': 'v2 per-task thresholds (audit.json)',
        'test_evaluated': False, 'training_graph': 'train proteins only',
        'validation_graph': 'train+validation proteins'}, indent=2))
    np.savez_compressed(run/'validation_predictions.npz', proteins=data['proteins'][valid],
                        predictions=best_prediction, labels=yt, mask=mt)


if __name__ == '__main__':
    main()
