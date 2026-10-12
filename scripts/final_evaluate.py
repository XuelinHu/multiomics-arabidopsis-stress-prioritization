"""Final protocol: train+validation refit, proteome prediction, test evaluation.

- Configuration: ModalityGraph, ESM-PCA256, control expression, filtered
  domains, and PPI>=700.
- Scaling, PCA, and domain vocabulary are fitted on train+validation only.
- Fixed epoch count is the median best epoch across three development seeds.
- Per-seed outputs are written to results/corrected; --aggregate summarizes them.
"""
import argparse
import json
import os
import random
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', PROJECT_ROOT/'data'))
RESULTS = Path('results')
SEEDS = (17, 42, 2026)
CORRECTED = RESULTS/'corrected'


def aggregate():
    directory = BASE/'processed/protein_inputs'
    data = np.load(directory/'inputs.npz')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    test = np.flatnonzero(split.split == 'test')
    paths = [CORRECTED/f'seed_{seed}_predictions.npz' for seed in SEEDS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing seed predictions: {missing}')
    predictions = []
    for seed, path in zip(SEEDS, paths):
        item = np.load(path)
        assert int(item['seed']) == seed
        assert np.array_equal(item['proteins'], data['proteins'])
        predictions.append(item['predictions'])
    preds = np.mean(predictions, axis=0).astype('float32')
    np.savez_compressed(CORRECTED/'final_predictions.npz', proteins=data['proteins'],
                        predictions=preds, labels=data['labels'], mask=data['label_mask'],
                        split=split.split.to_numpy())
    metrics = {}
    rng = np.random.default_rng(20260912)
    test_clusters = split.cluster.iloc[test].to_numpy()
    unique_clusters = np.unique(test_clusters)
    bootstrap_indices = []
    members = {cluster: test[test_clusters == cluster] for cluster in unique_clusters}
    for _ in range(2000):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        bootstrap_indices.append(np.concatenate([members[cluster] for cluster in sampled]))
    for t, task in enumerate(['salt', 'mannitol_osmotic', 'heat']):
        mask = data['label_mask'][test, t]
        y = data['labels'][test, t][mask]
        p = preds[test, t][mask]
        ap_boot, roc_boot = [], []
        for sampled in bootstrap_indices:
            sampled_mask = data['label_mask'][sampled, t]
            sampled_y = data['labels'][sampled, t][sampled_mask]
            if len(np.unique(sampled_y)) < 2:
                continue
            sampled_p = preds[sampled, t][sampled_mask]
            ap_boot.append(average_precision_score(sampled_y, sampled_p))
            roc_boot.append(roc_auc_score(sampled_y, sampled_p))
        metrics[task] = {
            'auprc': average_precision_score(y, p),
            'auprc_cluster_bootstrap_ci95': np.quantile(ap_boot, [.025, .975]).tolist(),
            'auroc': roc_auc_score(y, p),
            'auroc_cluster_bootstrap_ci95': np.quantile(roc_boot, [.025, .975]).tolist(),
            'n': len(y), 'prevalence': float(y.mean()),
        }
    metrics['macro_auprc'] = float(np.mean([metrics[t]['auprc'] for t in
                                            ['salt', 'mannitol_osmotic', 'heat']]))
    payload = {
        'protocol': ('Corrected train+validation indexing; three-seed probability average; '
                     'standardization/PCA/domain vocabulary fit on train+validation only'),
        'model': 'ModalityGraph (ESM-PCA256 + control expression + filtered domains + PPI>=700)',
        'fixed_epochs': 20, 'seeds': list(SEEDS),
        'uncertainty': '2000 test homology-cluster bootstrap replicates', 'test': metrics,
    }
    (CORRECTED/'test_metrics.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--deployment-only', action='store_true',
                        help='Retrain fixed protocol and save weights/preprocessing without test inference')
    args = parser.parse_args()
    if args.aggregate and args.deployment_only:
        parser.error('--aggregate cannot be combined with --deployment-only')
    if args.aggregate:
        aggregate()
        return
    seeds = (args.seed,) if args.seed is not None else SEEDS
    CORRECTED.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = BASE/'processed/protein_inputs'
    data = np.load(directory/'inputs.npz')
    proteins = pd.read_csv(directory/'proteins.tsv', sep='\t').fillna('')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    assert np.array_equal(data['proteins'], split.protein) and np.array_equal(data['proteins'], proteins.protein)
    train = np.flatnonzero(split.split == 'train')
    valid = np.flatnonzero(split.split == 'validation')
    test = np.flatnonzero(split.split == 'test')
    fit = np.concatenate([train, valid])   # Final fit set.
    nfit = len(fit)
    used = np.arange(len(data['proteins']))  # Full-proteome inference graph.
    lookup = {f'3702.{p}': i for i, p in enumerate(data['proteins'])}
    assert not np.intersect1d(fit, test).size
    assert len(np.unique(fit)) == nfit

    # Fit features on train+validation.
    esm = np.load(directory/'esm_embeddings.npz')
    assert np.array_equal(esm['proteins'], data['proteins'])
    esm_x = esm['embeddings'].astype('float32')
    s1 = StandardScaler().fit(esm_x[fit])
    pca = PCA(n_components=256, random_state=0).fit(s1.transform(esm_x[fit]))
    seq = pca.transform(s1.transform(esm_x)).astype('float32')
    s2 = StandardScaler().fit(data['control_expression'][fit])
    expr = s2.transform(data['control_expression']).astype('float32')
    domains = [set(filter(None, s.split(';'))) for s in proteins.InterPro]
    from collections import Counter
    freq = Counter()
    for i in fit:
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
    domain_matrix = np.concatenate([rows, counts, flags], axis=1).astype('float32')
    mods = [torch.tensor(seq, device=device),
            torch.tensor(expr, device=device),
            torch.tensor(domain_matrix, device=device)]

    # Full-proteome PPI inference graph.
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
    fit_graph = normalize(adjacency[fit][:, fit])
    full_graph = normalize(adjacency)

    # Fixed epoch count: median development best epoch for this configuration.
    best_epochs = []
    for s in SEEDS:
        h = pd.read_csv(f'results/development/modality_graph_seed{s}_esm_v2_domf/history.csv')
        best_epochs.append(int(h.loc[h.validation_macro_auprc.idxmax(), 'epoch']))
    epochs = int(np.ceil(np.median(best_epochs)))
    print(f'Development best epochs: {best_epochs} -> fixed {epochs} epochs', flush=True)

    fit_labels = torch.tensor(data['labels'][fit], device=device)
    fit_masks = torch.tensor(data['label_mask'][fit], device=device)
    positive = (fit_labels * fit_masks).sum(0)
    negative = ((1 - fit_labels) * fit_masks).sum(0)
    assert (positive > 0).all() and (negative > 0).all()

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ModalityGraph([x.shape[1] for x in mods], use_graph=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits, _ = model([x[fit] for x in mods], fit_graph)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, fit_labels, pos_weight=negative/positive, reduction='none')
            loss = (losses * fit_masks).sum()/fit_masks.sum()
            assert torch.isfinite(loss), f'seed {seed} epoch {epoch}: nonfinite loss'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save({
            'model': 'ModalityGraph', 'state_dict': state, 'seed': seed,
            'epochs': epochs, 'input_dims': [int(x.shape[1]) for x in mods],
            'tasks': ['salt', 'mannitol_osmotic', 'heat'],
        }, CORRECTED/f'seed_{seed}_deployment_model.pt')
        np.savez_compressed(
            CORRECTED/f'seed_{seed}_preprocessing.npz',
            esm_mean=s1.mean_.astype('float32'), esm_scale=s1.scale_.astype('float32'),
            pca_components=pca.components_.astype('float32'), pca_mean=pca.mean_.astype('float32'),
            expression_mean=s2.mean_.astype('float32'), expression_scale=s2.scale_.astype('float32'),
            domain_vocabulary=np.asarray(keep, dtype=str),
        )
        if args.deployment_only:
            print(f'seed {seed} deployment artifacts saved; test inference skipped', flush=True)
            continue
        model.eval()
        with torch.no_grad():
            p = model(mods, full_graph)[0].sigmoid().cpu().numpy()
        output = CORRECTED/f'seed_{seed}_predictions.npz'
        np.savez_compressed(output, proteins=data['proteins'], predictions=p.astype('float32'),
                            seed=np.array(seed), epochs=np.array(epochs))
        print(f'seed {seed} complete (final loss {loss.item():.4f})', flush=True)


if __name__ == '__main__':
    main()
