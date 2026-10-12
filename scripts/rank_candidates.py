"""未表征蛋白候选排序：预测概率 + 多源证据表。

注释覆盖度定义（依据本研究使用的本地 GO 与 InterPro 记录）：
  tier1 按本文标准无注释: GO==0 且 InterPro==0
  tier2 稀疏注释: GO<=2 且 InterPro<=1，排除 tier1
证据列：预测概率(S/M/H)、对照表达、PPI 度、PPI 邻域阳性比例、
        同源簇大小与簇内已注释成员数、域/GO 数、in_string。
输出 results/candidate_proteins.csv；表述使用"候选功能关联"。
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
RESULTS = Path('results')
TASKS = ['S', 'M', 'H']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    results = args.results_dir
    directory = BASE/'processed/protein_inputs'
    data = np.load(directory/'inputs.npz')
    proteins = pd.read_csv(directory/'proteins.tsv', sep='\t').fillna('')
    split = pd.read_csv(directory/'split.tsv', sep='\t')
    pred = np.load(results/'final_predictions.npz')
    assert np.array_equal(pred['proteins'], data['proteins'])
    P = pred['predictions']
    n = len(P)

    go_terms = {}
    for line in open(BASE/'raw/go/uniprot_arabidopsis_go.tsv'):
        if line.startswith('Entry'):
            continue
        e, _, gos = line.rstrip('\n').partition('\t')
        go_terms[e] = set(filter(None, gos.split('; ')))
    go_counts = np.array([len(go_terms.get(p, set())) for p in data['proteins']])
    dom_counts = np.array([len(set(filter(None, s.split(';')))) for s in proteins.InterPro])
    tier1 = (go_counts == 0) & (dom_counts == 0)
    tier2 = (go_counts <= 2) & (dom_counts <= 1) & ~tier1

    # PPI 度与邻域证据
    lookup = {f'3702.{p}': i for i, p in enumerate(data['proteins'])}
    rr, cc = [], []
    for chunk in pd.read_csv(BASE/'raw/string/3702.protein.links.v12.0.txt.gz', sep=' ', chunksize=200000):
        chunk = chunk[chunk.combined_score >= 700]
        for a, b, w in chunk.itertuples(index=False, name=None):
            i, j = lookup.get(a), lookup.get(b)
            if i is None or j is None or i == j:
                continue
            rr.append(i)
            cc.append(j)
    adj = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n))
    adj = adj.maximum(adj.T)
    degree = np.asarray(adj.sum(1)).ravel()
    labels = data['labels']
    masks = data['label_mask']
    evidence_pool = split.split.isin(['train', 'validation']).to_numpy()
    neigh_pos = np.full((n, 3), np.nan)
    for i in range(n):
        row = adj.getrow(i).indices
        for task in range(3):
            observed = row[evidence_pool[row] & masks[row, task]]
            if len(observed):
                neigh_pos[i, task] = labels[observed, task].mean()

    # 同源簇：protein -> representative
    cluster = pd.read_csv(directory/'homology_cluster.tsv', sep='\t', header=None, names=['protein', 'rep'])
    sizes = cluster.groupby('rep').size()
    rep_of = dict(zip(cluster.protein, cluster.rep))
    prot_idx = {p: i for i, p in enumerate(data['proteins'])}
    cluster['annotated'] = cluster.protein.map(lambda p: int(go_counts[prot_idx[p]] > 0))
    ann_members = cluster.groupby('rep')['annotated'].sum()

    rows = []
    for tier, idxs, label in [(1, np.flatnonzero(tier1), 'tier1_unannotated'),
                              (2, np.flatnonzero(tier2), 'tier2_sparse')]:
        for i in idxs:
            p = data['proteins'][i]
            rows.append({
                'protein': p, 'gene': data['genes'][i], 'tier': label,
                'split': split.split.iloc[i],
                'pred_salt': round(float(P[i, 0]), 4),
                'pred_mannitol': round(float(P[i, 1]), 4),
                'pred_heat': round(float(P[i, 2]), 4),
                'max_prob': round(float(P[i].max()), 4),
                'control_expr_mean': round(float(data['control_expression'][i].mean()), 3),
                'ppi_degree': int(degree[i]),
                'neighbor_pos_S': round(float(neigh_pos[i, 0]), 3),
                'neighbor_pos_M': round(float(neigh_pos[i, 1]), 3),
                'neighbor_pos_H': round(float(neigh_pos[i, 2]), 3),
                'cluster_size': int(sizes.get(rep_of.get(p, p), 1)),
                'cluster_annotated_members': int(ann_members.get(rep_of.get(p, p), int(go_counts[i] > 0))),
                'go_count': int(go_counts[i]),
                'domain_count': int(dom_counts[i]),
                'in_string': bool(proteins.in_string.iloc[i]),
            })
    out = pd.DataFrame(rows).sort_values('max_prob', ascending=False)
    out.to_csv(results/'candidate_proteins.csv', index=False)
    print(f'候选蛋白: tier1={len(out[out.tier=="tier1_unannotated"])}, tier2={len(out[out.tier=="tier2_sparse"])}')
    print('Top-10 候选:')
    print(out.head(10)[['protein', 'gene', 'tier', 'pred_salt', 'pred_mannitol', 'pred_heat',
                        'ppi_degree', 'cluster_annotated_members', 'go_count', 'domain_count']].to_string(index=False))
    print(f"已写入 {results/'candidate_proteins.csv'}")


if __name__ == '__main__':
    main()
