"""Align measured genes to proteins; keep uncertain response labels masked."""
import json
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
OUT = BASE/'processed/protein_inputs'

def main():
    source = BASE/'raw/uniprot/mapping_snapshot'
    assert (source/'verified.json').exists(), 'Complete mapping snapshot required'
    mapping = pd.read_csv(source/'proteome_mapping.tsv',sep='\t').fillna('')
    expression = pd.read_csv(BASE/'processed/GSE146206/gene_tpm.tsv.gz',sep='\t',index_col=0)
    differential = pd.read_csv(BASE/'processed/GSE146206/author_differential_results.tsv.gz',sep='\t')
    sequences = {}
    current = None
    for line in open(BASE/'raw/uniprot/UP000006548_proteome.fasta'):
        if line.startswith('>'):
            current = line.split('|')[1]
            assert current not in sequences
            sequences[current] = ''
        else:
            sequences[current] += line.strip()
    info = pd.read_csv(BASE/'raw/string/3702.protein.info.v12.0.txt.gz',sep='\t')
    string_ids = set(info.iloc[:,0].str.removeprefix('3702.'))
    links, rejected = [], []
    for _, row in mapping.iterrows():
        loci = sorted(set(re.findall(r'AT[1-5CM]G\d{5}',(row['TAIR']+' '+row['Gene Names (ordered locus)']).upper())))
        reason = None
        if len(loci)!=1: reason='ambiguous_or_missing_locus'
        elif loci[0] not in expression.index: reason='not_measured'
        elif row.Entry not in sequences: reason='missing_sequence'
        if reason:
            rejected.append({'protein':row.Entry,'reason':reason}); continue
        links.append({'protein':row.Entry,'gene':loci[0],'length':len(sequences[row.Entry]),
                      'in_string':row.Entry in string_ids,'InterPro':row.InterPro,'Pfam':row.Pfam})
    candidates = pd.DataFrame(links)
    # Stable representative per gene: prefer PPI coverage, then longest sequence, then accession.
    selected = candidates.sort_values(['gene','in_string','length','protein'],ascending=[True,False,False,True]).drop_duplicates('gene').reset_index(drop=True)
    OUT.mkdir(parents=True,exist_ok=True)
    candidates.to_csv(OUT/'all_mappings.tsv',sep='\t',index=False)
    selected.to_csv(OUT/'proteins.tsv',sep='\t',index=False)
    pd.DataFrame(rejected).to_csv(OUT/'unmapped.tsv',sep='\t',index=False)
    with (OUT/'representatives.fasta').open('w') as handle:
        for row in selected.itertuples(): handle.write(f'>{row.protein}\n{sequences[row.protein]}\n')
    # Predict differential-response evidence, not physiological causation or validated function.
    diff = differential.set_index('AGI').loc[selected.gene]
    labels, masks = [], []
    report = {}
    for task in ['S','M','H']:
        q = diff['qvalue_'+task].to_numpy(float)
        beta = diff['b_'+task].to_numpy(float)
        positive = np.isfinite(q)&np.isfinite(beta)&(q<=.05)
        comparator = np.isfinite(q)&(q>=.5)
        labels.append(positive.astype(float)); masks.append(positive|comparator)
        report[task]={'positive':int(positive.sum()),'nondifferential_comparator':int(comparator.sum()),'uncertain':int((~(positive|comparator)).sum())}
    # Only untreated expression enters this task. No treatment values, GO outcomes, or response edges.
    background = np.log1p(expression.loc[selected.gene,['C_1','C_2','C_3']].to_numpy(float))
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    composition = np.array([[sequences[p].count(a)/len(sequences[p]) for a in aa]+[np.log1p(len(sequences[p]))] for p in selected.protein])
    np.savez_compressed(OUT/'inputs.npz', sequence_composition=composition.astype('float32'),
                        control_expression=background.astype('float32'),labels=np.array(labels,dtype='float32').T,
                        label_mask=np.array(masks).T,proteins=selected.protein.to_numpy(dtype=str),genes=selected.gene.to_numpy(dtype=str))
    (OUT/'audit.json').write_text(json.dumps({'mapped_proteins':len(candidates),'representative_genes':len(selected),
        'with_string':int(selected.in_string.sum()),'labels':report,
        'limitation':'q>=0.5 is a nondifferential comparator, not verified biological negative; q thresholds require sensitivity analysis. M is mannitol/osmotic stress. Sequence composition is a baseline, not ESM.'},indent=2))
    print((OUT/'audit.json').read_text())

if __name__ == '__main__': main()
