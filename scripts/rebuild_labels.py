#!/usr/bin/env python3
"""重建 S/M/H 标签：q/b 阈值敏感性分析 + 应用选定阈值。

阳性 = 有限(q) 且 有限(b) 且 q<=Q 且 |b|>=B；对照 = 有限(q) 且 q>=0.5；其余掩码。
- 默认只做敏感性分析，输出 results/label_sensitivity.csv；
- --q Q --b B --apply 时把新标签写回 inputs.npz（首次自动备份原文件）。
"""
import argparse
import json
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
OUT = BASE/'processed/protein_inputs'
RESULTS = Path(__file__).resolve().parents[1] / 'results'

def load():
    proteins = pd.read_csv(OUT/'proteins.tsv', sep='\t')
    diff = pd.read_csv(BASE/'processed/GSE146206/author_differential_results.tsv.gz', sep='\t')
    diff = diff.set_index('AGI').loc[proteins.gene]
    q = np.stack([diff[f'qvalue_{t}'].to_numpy(float) for t in 'SMH'], axis=1)
    b = np.stack([diff[f'b_{t}'].to_numpy(float) for t in 'SMH'], axis=1)
    return proteins, q, b

def build(q, b, thresholds):
    """thresholds: {task: (Q, B)}；对照统一 q>=0.5"""
    positive = np.zeros(q.shape, bool)
    for i, t in enumerate('SMH'):
        Q, B = thresholds[t]
        positive[:, i] = np.isfinite(q[:, i]) & np.isfinite(b[:, i]) & (q[:, i] <= Q) & (np.abs(b[:, i]) >= B)
    comparator = np.isfinite(q) & (q >= 0.5)
    labels = positive.astype(float)
    masks = positive | comparator
    return labels, masks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q-s', type=float, default=0.05); ap.add_argument('--b-s', type=float, default=0.0)
    ap.add_argument('--q-m', type=float, default=0.05); ap.add_argument('--b-m', type=float, default=0.5)
    ap.add_argument('--q-h', type=float, default=0.001); ap.add_argument('--b-h', type=float, default=1.5)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    thresholds = {'S': (args.q_s, args.b_s), 'M': (args.q_m, args.b_m), 'H': (args.q_h, args.b_h)}
    proteins, q, b = load()
    n = len(proteins)
    tasks = ['S', 'M', 'H']
    # 敏感性分析（记录用；最终阈值按任务独立，见 thresholds）
    rows = []
    for Q in (0.05, 0.01, 0.001):
        for B in (0.0, 0.5, 1.0, 1.5, 2.0):
            labels, masks = build(q, b, {'S': (Q, B), 'M': (Q, B), 'H': (Q, B)})
            row = {'q': Q, 'b': B}
            for i, t in enumerate(tasks):
                npos = int(labels[:, i].sum()); nmask = int(masks[:, i].sum())
                row[f'{t}_positive'] = npos
                row[f'{t}_masked'] = nmask
                row[f'{t}_prevalence'] = round(npos / max(nmask, 1), 4)
            rows.append(row)
    labels, masks = build(q, b, thresholds)
    chosen = {'q': '', 'b': ''}
    for i, t in enumerate(tasks):
        npos = int(labels[:, i].sum()); nmask = int(masks[:, i].sum())
        chosen[f'{t}_positive'] = npos
        chosen[f'{t}_masked'] = nmask
        chosen[f'{t}_prevalence'] = round(npos / max(nmask, 1), 4)
        chosen[f'{t}_q'] = thresholds[t][0]
        chosen[f'{t}_b'] = thresholds[t][1]
    rows.append(chosen)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS/'label_sensitivity.csv', index=False)
    print('敏感性分析(统一阈值) + 最后一行=选定任务独立阈值:')
    print(pd.DataFrame(rows).to_string(index=False))
    print('sensitivity 表已写入 results/label_sensitivity.csv')
    # 应用阈值
    if args.apply:
        npz = OUT/'inputs.npz'
        backup = OUT/'inputs_original_labels_v0.npz'
        if not backup.exists():
            shutil.copy2(npz, backup)
            print('已备份原标签 ->', backup)
        data = np.load(npz, allow_pickle=True)
        np.savez_compressed(npz, sequence_composition=data['sequence_composition'],
                            control_expression=data['control_expression'],
                            labels=labels.astype('float32'), label_mask=masks,
                            proteins=data['proteins'], genes=data['genes'])
        report = {}
        for i, t in enumerate(tasks):
            report[t] = {'positive': int(labels[:, i].sum()),
                         'nondifferential_comparator': int(masks[:, i].sum()) - int(labels[:, i].sum()),
                         'uncertain': int(n - masks[:, i].sum())}
        audit = json.load(open(OUT/'audit.json'))
        audit['label_thresholds'] = {t: {'q': thresholds[t][0], 'b': thresholds[t][1]} for t in tasks}
        audit['labels'] = report
        audit['limitation'] = ('labels rebuilt 2026-09-11: per-task thresholds (S q<=0.05,|b|>=0; '
                               'M q<=0.05,|b|>=0.5; H q<=0.001,|b|>=1.5) chosen by prevalence 5-60% and >=200 '
                               'positives, fixed before model training on these labels; comparator=q>=0.5 is a '
                               'nondifferential comparator, not verified biological negative; heat response is '
                               'broad (97% DE at q<=0.05) hence the effect-size floor.')
        json.dump(audit, open(OUT/'audit.json', 'w'), indent=2, ensure_ascii=False)
        print('已应用任务独立阈值 ->', report)

if __name__ == '__main__':
    main()
