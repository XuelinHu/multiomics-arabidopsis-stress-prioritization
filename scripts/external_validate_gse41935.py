#!/usr/bin/env python3
"""GSE41935 跨平台验证：Col 生态型叶片 Ctrl vs Salt / Ctrl vs Heat（NimbleGen 微阵列）。

- 解析 series matrix（行为 AT 基因 ID），按标题 Leaf-{Eco}-{Cond}-rep 抽样本
- 每基因 log2FC(Salt/Ctrl) 与 log2FC(Heat/Ctrl)，与最终预测做 Spearman + AUROC(top20%)
- 该数据集不参与任何训练/特征选择
"""
import argparse, gzip, json, math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('KGMULTIOMICS_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data'))
MATRIX = f"{BASE}/raw/geo/GSE41935/GSE41935_series_matrix.txt.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    results = args.results_dir
    # 样本标题
    ts = []
    for line in gzip.open(MATRIX, "rt"):
        if line.startswith("!Sample_title"):
            ts = [t.strip('"') for t in line.rstrip().split("\t")[1:]]
            break
    accs = []
    for line in gzip.open(MATRIX, "rt"):
        if line.startswith("!Sample_geo_accession"):
            accs = [a.strip('"') for a in line.rstrip().split("\t")[1:]]
            break
    sel = {"Ctrl": [], "Salt": [], "Heat": []}
    for a, t in zip(accs, ts):
        parts = t.split("-")
        if len(parts) >= 4 and parts[0] == "Leaf" and parts[1] == "Col":
            cond = parts[2]
            if cond in sel:
                sel[cond].append(a)  # 每条件全部重复
    print("Col 样本:", {c: len(v) for c, v in sel.items()})
    mat = pd.read_csv(MATRIX, sep="\t", comment="!", index_col=0)
    print("矩阵:", mat.shape)
    used = []
    for c, gsms in sel.items():
        if gsms:
            mat[c] = mat[gsms].mean(axis=1)
            used.append(c)
    sub = mat[used]
    sub.index = [g.split(".")[0] for g in sub.index]  # 行即 AT 基因 ID
    agg = sub.groupby(sub.index).median()
    out = {}
    for cond in ("Salt", "Heat"):
        if cond not in agg.columns:
            continue
        fc = (agg[cond] - agg["Ctrl"]).to_dict()  # 表达已 log2
        genes = list(fc)
        inputs = np.load(f"{BASE}/processed/protein_inputs/inputs.npz", allow_pickle=True)
        pred = np.load(results/'final_predictions.npz', allow_pickle=True)
        g2i = {g: i for i, g in enumerate(inputs["genes"])}
        P = pred["predictions"]
        rows = []
        for g in genes:
            if g in g2i and math.isfinite(fc[g]):
                rows.append((g, fc[g], g2i[g]))
        if len(rows) < 100:
            print(f"{cond}: 映射基因不足 ({len(rows)})，跳过")
            continue
        fcs = np.array([r[1] for r in rows])
        idxs = [r[2] for r in rows]
        res = {"n_genes": len(rows), "contrast": f"Col {cond} vs Ctrl"}
        for t, task in enumerate(["salt", "mannitol_osmotic", "heat"]):
            p = P[idxs, t]
            rho, pval = spearmanr(p, fcs)
            abs_rho, abs_pval = spearmanr(p, np.abs(fcs))
            thr = np.quantile(fcs, 0.8)
            y = (fcs >= thr).astype(int)
            auroc = roc_auc_score(y, p)
            rng = np.random.default_rng(42)
            boots = [roc_auc_score(y[b], p[b]) for b in rng.integers(0, len(y), size=(1000, len(y)))]
            lo, hi = np.quantile(boots, [0.025, 0.975])
            abs_y = (np.abs(fcs) >= np.quantile(np.abs(fcs), 0.8)).astype(int)
            abs_auroc = roc_auc_score(abs_y, p)
            res[task] = {"spearman_rho": float(rho), "spearman_p": float(pval),
                         "auroc_top20": float(auroc), "auroc_ci95": [float(lo), float(hi)],
                         "absolute_response_spearman_rho": float(abs_rho),
                         "absolute_response_spearman_p": float(abs_pval),
                         "auroc_top20_absolute_response": float(abs_auroc)}
            print(f"{cond} {task}: rho={rho:.3f} AUROC={auroc:.3f} CI[{lo:.3f},{hi:.3f}]")
        out[cond] = res
    with open(results/'external_gse41935_validation.json', 'w') as handle:
        json.dump(out, handle, indent=2)
    print(f"已写入 {results/'external_gse41935_validation.json'}")


if __name__ == "__main__":
    main()
