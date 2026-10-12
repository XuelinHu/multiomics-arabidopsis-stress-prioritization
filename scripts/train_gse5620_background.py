#!/usr/bin/env python3
"""GSE5620 (AtGenExpress 非生物胁迫对照系列, 36 样本) 背景表达增强。

- GSE5620 下载子集全部为 Control-Shoots/Roots 时间点样本（无处理样本，无标签泄漏）
- GPL198 探针映射到 AT 基因（median 聚合），36 维对照表达作为第 4 模态
- 协议与主实验一致；无背景表达的对照 = modality_graph_*_v2_domf 已有结果
"""
import argparse
import copy
import gzip
import json
import os
import random
import sys
import urllib.request
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
GSE5620 = f"{BASE}/raw/geo/GSE5620_series_matrix.txt.gz"
GPL198 = f"{BASE}/raw/geo/GPL198.annot.gz"


def ensure_gpl198():
    if not os.path.exists(GPL198):
        # FTP 无 annot 文件；改从 GEO 平台页面取表（含 AGI 列）
        url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL198&targ=self&form=text&view=data"
        print("下载 GPL198 平台表 ...")
        urllib.request.urlretrieve(url, "/tmp/gpl198.txt")
        with open("/tmp/gpl198.txt") as f, open(GPL198, "w") as out:
            in_table = False
            for line in f:
                if line.startswith("!platform_table_begin"):
                    in_table = True
                    continue
                if line.startswith("!platform_table_end"):
                    break
                if in_table:
                    out.write(line)
    return GPL198


def background_matrix():
    """返回 DataFrame: index=AT 基因, columns=36 个对照样本（log2 表达已标准化）"""
    ensure_gpl198()
    mat = pd.read_csv(GSE5620, sep="\t", comment="!", index_col=0)
    # 探针 -> 基因（GPL198 表的 AGI 列）
    header = open(GPL198).readline().rstrip("\n").split("\t")
    gene_col = header.index("AGI") if "AGI" in header else None
    assert gene_col is not None, "GPL198 无 AGI 列"
    probe2gene = {}
    for line in open(GPL198):
        if line.startswith("ID"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= gene_col:
            continue
        sym = parts[gene_col].strip()
        if sym.startswith("AT") and len(sym) <= 10:
            probe2gene[parts[0].strip()] = sym
    mat["gene"] = [probe2gene.get(p, "") for p in mat.index]
    mat = mat[mat.gene != ""]
    agg = mat.groupby("gene").median()
    print(f"GSE5620 背景矩阵: {agg.shape[0]} 基因 x {agg.shape[1]} 样本（全部对照）")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--run-suffix', default='_v2_bkg')
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
    bkg = background_matrix().reindex(proteins.gene).to_numpy(float)
    bkg = np.where(np.isfinite(bkg), bkg, 0.0)
    # 特征
    esm = np.load(directory/'esm_embeddings.npz')
    esm_x = esm['embeddings'].astype('float32')
    s1 = StandardScaler().fit(esm_x[train])
    pca = PCA(n_components=256, random_state=args.seed).fit(s1.transform(esm_x[train]))
    seq = pca.transform(s1.transform(esm_x[used])).astype('float32')
    s2 = StandardScaler().fit(data['control_expression'][train])
    expr = s2.transform(data['control_expression'][used]).astype('float32')
    s3 = StandardScaler().fit(bkg[train])
    expr_bkg = s3.transform(bkg[used]).astype('float32')
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
               torch.tensor(dom[used], device=device), torch.tensor(expr_bkg, device=device)]
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
    labels = torch.tensor(data['labels'][used], device=device)
    masks = torch.tensor(data['label_mask'][used], device=device)
    positive = (labels[:ntrain] * masks[:ntrain]).sum(0)
    negative = ((1 - labels[:ntrain]) * masks[:ntrain]).sum(0)
    model = ModalityGraph([x.shape[1] for x in tensors], use_graph=True).to(device)
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
            logits, labels[:ntrain], pos_weight=negative/positive, reduction='none')
        loss = (losses * masks[:ntrain]).sum()/masks[:ntrain].sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            p = model(tensors, val_graph)[0][ntrain:].sigmoid().cpu().numpy()
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
        'seed': args.seed,
        'features': 'ESM-PCA256 + control expression + filtered domains + GSE5620 36-sample control background',
        'note': 'GSE5620 subset contains control samples only (no treatment leakage)',
        'test_evaluated': False}, indent=2))


if __name__ == '__main__':
    main()
