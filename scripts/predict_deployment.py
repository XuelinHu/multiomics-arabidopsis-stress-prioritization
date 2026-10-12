"""Predict with fixed deployment weights and precomputed aligned features.

No response labels or label masks are read, and no evaluation is performed.
Supply saved deployment weights and aligned input features before inference.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.modality_graph import ModalityGraph
from src.data.alignment import align_protein_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('KGMULTIOMICS_DATA_ROOT',ROOT/'data')))
    parser.add_argument('--weights-dir',type=Path,default=ROOT/'results/corrected')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    directory = args.data_root/'processed/protein_inputs'
    with np.load(directory/'inputs.npz') as data:
        ids = data['proteins']
        expression = data['control_expression'].astype('float32')
        expression = align_protein_arrays(ids,expression,ids)
    with np.load(directory/'esm_embeddings.npz') as data:
        esm = align_protein_arrays(data['proteins'],data['embeddings'].astype('float32'),ids)
    proteins = pd.read_csv(directory/'proteins.tsv',sep='\t').fillna('')
    if proteins.protein.duplicated().any():raise ValueError('Duplicate metadata protein identifiers')
    if not set(ids).issubset(set(proteins.protein)):raise ValueError('Required protein metadata missing')
    proteins=proteins.set_index('protein').loc[ids].reset_index()
    domains = [set(filter(None,value.split(';'))) for value in proteins.InterPro]
    lookup = {f'3702.{identifier}':i for i,identifier in enumerate(ids)}
    rows,cols,weights = [],[],[]
    for chunk in pd.read_csv(args.data_root/'raw/string/3702.protein.links.v12.0.txt.gz',sep=r'\s+',chunksize=200000):
        for a,b,w in chunk.loc[chunk.combined_score>=700,['protein1','protein2','combined_score']].itertuples(index=False,name=None):
            i,j = lookup.get(a),lookup.get(b)
            if i is not None and j is not None and i!=j:
                rows.append(i); cols.append(j); weights.append(w/1000)
    adjacency = sparse.csr_matrix((weights,(rows,cols)),shape=(len(ids),len(ids)),dtype='float32')
    adjacency = adjacency.maximum(adjacency.T)+sparse.eye(len(ids),dtype='float32')
    adjacency = (sparse.diags(1/np.asarray(adjacency.sum(1)).ravel())@adjacency).tocoo()
    graph = torch.sparse_coo_tensor(np.stack([adjacency.row,adjacency.col]),adjacency.data,
                                    adjacency.shape,device=device).coalesce()
    predictions = []
    for seed in (17,42,2026):
        state = torch.load(args.weights_dir/f'seed_{seed}_deployment_model.pt',map_location='cpu',weights_only=True)
        with np.load(args.weights_dir/f'seed_{seed}_preprocessing.npz') as prep:
            seq = (((esm-prep['esm_mean'])/prep['esm_scale']-prep['pca_mean'])@prep['pca_components'].T).astype('float32')
            expr = ((expression-prep['expression_mean'])/prep['expression_scale']).astype('float32')
            vocabulary = prep['domain_vocabulary'].tolist()
        columns = {term:i for i,term in enumerate(vocabulary)}
        domain = np.zeros((len(ids),len(columns)+2),dtype='float32')
        for i,terms in enumerate(domains):
            matching = terms & columns.keys()
            for term in matching:
                domain[i,columns[term]] = 1
            domain[i,-2] = np.log1p(len(matching)); domain[i,-1] = bool(terms)
        modalities = [torch.tensor(x,device=device) for x in (seq,expr,domain)]
        if [x.shape[1] for x in modalities] != state['input_dims']:
            raise ValueError('Deployment input dimensions do not match weights')
        model = ModalityGraph(state['input_dims'],use_graph=True).to(device)
        model.load_state_dict(state['state_dict']); model.eval()
        with torch.no_grad():
            predictions.append(model(modalities,graph)[0].sigmoid().cpu().numpy())
    values = np.mean(predictions,axis=0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite deployment predictions')
    table = pd.DataFrame(values,columns=['salt','mannitol_osmotic','heat'])
    table.insert(0,'protein',ids)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output,index=False)
    print(f'Saved {len(table)} protein scores; no response labels were read')


if __name__ == '__main__':
    main()
