#!/usr/bin/env python3
"""用 ESM-2 t33_650M 预计算拟南芥蛋白嵌入（冻结权重，仅前向）。

- 输入：UniProt proteome FASTA + inputs.npz 的 27,322 个代表蛋白
- 输出：data/processed/protein_inputs/esm_embeddings.npz
    embeddings: (n, 1280) float32，proteins: (n,) 蛋白 ID
- 长序列（>1022 aa）按 1020aa 窗口、步长 510 滑动，加权平均；
- fp16 autocast 加速，输出保存为 float32。
"""
import os, time, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

DATA_ROOT = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
CKPT = DATA_ROOT / 'raw/esm/esm2_t33_650M_UR50D'
FASTA = DATA_ROOT / 'raw/uniprot/UP000006548_proteome.fasta'
NPZ = DATA_ROOT / 'processed/protein_inputs/inputs.npz'
OUT = DATA_ROOT / 'processed/protein_inputs/esm_embeddings.npz'

MAXLEN = 1022  # 不含 BOS/EOS；模型上限 1024
WIN, STRIDE = 1020, 510  # 长序列滑动窗口

def read_fasta(path):
    seqs, cur_id, cur = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if cur_id:
                seqs[cur_id] = "".join(cur)
            cur_id = line[1:].split("|")[1]  # sp|ACC|NAME -> ACC
            cur = []
        else:
            cur.append(line.strip())
    if cur_id:
        seqs[cur_id] = "".join(cur)
    return seqs

def windows(seq):
    """长序列分窗，返回 [(start, end)]，end 为绝对残基位置（python 切片）"""
    if len(seq) <= MAXLEN:
        return [(0, len(seq))]
    ws = []
    s = 0
    while s < len(seq):
        e = min(s + WIN, len(seq))
        ws.append((s, e))
        if e == len(seq):
            break
        s += STRIDE
        if s >= len(seq):  # 兜底：最后一个窗口对齐到末尾
            s = len(seq) - WIN
            if (s, len(seq)) in ws:
                break
    # 去重并排序
    return sorted(set(ws))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-proteins", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    protein_ids = list(d["proteins"][: args.max_proteins] if args.max_proteins else d["proteins"])
    seqs = read_fasta(FASTA)
    missing = [p for p in protein_ids if p not in seqs]
    assert not missing, f"FASTA 缺少 {len(missing)} 个蛋白: {missing[:5]}"
    seqs = {p: seqs[p].rstrip("*") for p in protein_ids}

    tok = EsmTokenizer.from_pretrained(str(CKPT))
    model = EsmModel.from_pretrained(str(CKPT)).cuda().eval()
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    emb = np.zeros((len(protein_ids), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    n_done = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(protein_ids), args.batch_size):
            batch = protein_ids[i : i + args.batch_size]
            # 每个蛋白可能拆多个窗口
            flat_ids, flat_seqs, backref = [], [], []
            for p in batch:
                for (s, e) in windows(seqs[p]):
                    flat_ids.append(p)
                    flat_seqs.append(seqs[p][s:e])
                    backref.append(p)
            enc = tok(flat_seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc)
            h = out.last_hidden_state  # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, H)
            pooled = pooled.float().cpu().numpy()
            # 按蛋白聚合窗口（等权平均）
            from collections import defaultdict
            acc = defaultdict(list)
            for p, v in zip(backref, pooled):
                acc[p].append(v)
            for p in batch:
                emb[n_done] = np.mean(acc[p], axis=0)
                n_done += 1
            if (i // args.batch_size) % 10 == 0:
                rate = n_done / (time.time() - t0)
                print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(protein_ids)} 蛋白, {rate:.1f} 蛋白/s, "
                      f"预计剩余 {(len(protein_ids)-n_done)/rate/60:.1f} min", flush=True)

    np.savez(OUT, embeddings=emb, proteins=np.array(protein_ids))
    print(f"DONE {len(protein_ids)} 蛋白, 用时 {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
    # 快速 sanity check
    print("shape:", emb.shape, "| 有限值:", np.isfinite(emb).all(), "| L2 范数范围:",
          np.linalg.norm(emb, axis=1).min(), "-", np.linalg.norm(emb, axis=1).max(), flush=True)

if __name__ == "__main__":
    main()
