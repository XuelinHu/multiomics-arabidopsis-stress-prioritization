#!/usr/bin/env python3
"""External comparison with GSE220276 soil drought (Bulk1, 30% SWC).

The model uses GSE146206 salt, mannitol, and heat labels. Its final predictions
are compared with independent drought expression measurements:
- Bulk1 drought0 (D30, three replicates) vs well-watered time 0 (three).
- Spearman correlation with drought log2FC, AUROC for the top 20% induced
  genes, bootstrap 95% CI, and permutation-test p-value.
The 3v3 design limits power; report these results as exploratory evidence.
"""
import argparse, gzip, json, math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('KGMULTIOMICS_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    results = args.results_dir
    # GSE220276 Bulk1 counts: D30 (3) vs ww-0 (3).
    cols = {}
    with gzip.open(f"{BASE}/raw/geo/GSE220276/GSE220276_Bulk1_Raw_Counts.txt.gz", "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, c in enumerate(header[1:], start=1):
            cols[c] = i
    d30 = ["Bulk1_D30-1", "Bulk1_D30-2", "Bulk1_D30-3"]
    ww0 = ["Bulk1_ww-0-1", "Bulk1_ww-0-2", "Bulk1_ww-0-3"]
    genes, fc = [], []
    with gzip.open(f"{BASE}/raw/geo/GSE220276/GSE220276_Bulk1_Raw_Counts.txt.gz", "rt") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            g = parts[0]
            if not g.startswith("AT"):
                continue
            d = np.array([float(parts[cols[c]]) for c in d30])
            w = np.array([float(parts[cols[c]]) for c in ww0])
            genes.append(g)
            fc.append(math.log2((d.mean() + 1) / (w.mean() + 1)))
    fc = np.array(fc)
    print(f"GSE220276 Bulk1 genes: {len(genes)}")

    # Map genes to proteins and retrieve final predictions.
    inputs = np.load(f"{BASE}/processed/protein_inputs/inputs.npz", allow_pickle=True)
    pred = np.load(results/'final_predictions.npz', allow_pickle=True)
    gene2idx = {g: i for i, g in enumerate(inputs["genes"])}
    mapped = [(g, effect, gene2idx[g]) for g, effect in zip(genes, fc) if g in gene2idx]
    genes = [row[0] for row in mapped]
    fc = np.array([row[1] for row in mapped])
    idx = [row[2] for row in mapped]
    P = pred["predictions"]
    print(f"Mapped to proteome: {len(genes)} genes")

    out = {"n_genes": len(genes), "contrast": "drought 30% SWC time0 vs well-watered time0 (3v3)"}
    for t, task in enumerate(["salt", "mannitol_osmotic", "heat"]):
        p = P[idx, t]
        rho, pval = spearmanr(p, fc)
        abs_rho, abs_pval = spearmanr(p, np.abs(fc))
        # AUROC: top 20% drought-induced genes are positives.
        thr = np.quantile(fc, 0.8)
        y = (fc >= thr).astype(int)
        auroc = roc_auc_score(y, p) if y.sum() > 0 and (1 - y).sum() > 0 else float("nan")
        # bootstrap 95% CI
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(1000):
            b = rng.integers(0, len(y), len(y))
            boots.append(roc_auc_score(y[b], p[b]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        # Permutation test.
        perm = []
        for _ in range(1000):
            permuted = p.copy()
            rng.shuffle(permuted)
            perm.append(roc_auc_score(y, permuted))
        p_perm = float((np.array(perm) >= auroc).mean())
        abs_y = (np.abs(fc) >= np.quantile(np.abs(fc), 0.8)).astype(int)
        abs_auroc = roc_auc_score(abs_y, p)
        out[task] = {"spearman_rho": float(rho), "spearman_p": float(pval),
                     "auroc_top20_drought_induced": float(auroc), "auroc_ci95": [float(lo), float(hi)],
                     "permutation_p": p_perm, "absolute_response_spearman_rho": float(abs_rho),
                     "absolute_response_spearman_p": float(abs_pval),
                     "auroc_top20_absolute_response": float(abs_auroc)}
        print(f"{task}: rho={rho:.3f} (p={pval:.2e}) | AUROC(top 20% drought-induced)={auroc:.3f} CI[{lo:.3f},{hi:.3f}] permutation p={p_perm:.3f}")
    with open(results/'external_drought_validation.json', 'w') as handle:
        json.dump(out, handle, indent=2)
    print(f"Saved {results/'external_drought_validation.json'}")


if __name__ == "__main__":
    main()
