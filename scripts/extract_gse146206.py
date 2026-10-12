"""Extract author-provided quantification, preserving IDs and source provenance."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get('KGMULTIOMICS_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))
SOURCE = BASE / 'raw/geo/GSE146206/GSE146206_SupplementalDataset.xlsx'
OUT = BASE / 'processed/GSE146206'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SOURCE, header=1)
    assert df.AGI.notna().all() and df.target_id.notna().all()
    assert not df.target_id.duplicated().any(), 'Resolve duplicate target IDs before aggregation'
    counts = df.set_index('target_id').filter(regex='^est_counts_')
    counts.columns = counts.columns.str.replace('est_counts_', '', regex=False)
    tpm = df.set_index('target_id')[counts.columns]
    assert counts.shape[1] == 24
    for name, matrix in [('estimated_counts', counts), ('tpm', tpm)]:
        values = matrix.to_numpy(dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all(), name
        matrix.to_csv(OUT / f'transcript_{name}.tsv.gz', sep='\t')
        matrix.assign(AGI=df.set_index('target_id').AGI).groupby('AGI').sum().to_csv(
            OUT / f'gene_{name}.tsv.gz', sep='\t')
    df[['AGI', 'target_id']].to_csv(OUT / 'transcript_gene.tsv', sep='\t', index=False)
    diff = df[['AGI','target_id'] + [c for c in df if c.startswith(('qvalue_', 'b_'))]]
    diff.to_csv(OUT / 'author_differential_results.tsv.gz', sep='\t', index=False)
    # These outcomes must not also become predictors of their derived labels.
    treatments = {'Control':'C', 'Salt':'S', 'Mannitol':'M', 'Heat':'H',
                  'Salt_Mannitol':'SM', 'Salt_Heat':'SH', 'Mannitol_Heat':'MH',
                  'Salt_Mannitol_Heat':'SMH'}
    import gzip
    samples = []
    for line in gzip.open(BASE/'raw/geo/GSE146206/GSE146206_family.soft.gz', 'rt'):
        if line.startswith('^SAMPLE = '):
            samples.append({'gsm':line.strip().split(' = ')[1]})
        elif line.startswith('!Sample_characteristics_ch1 = '):
            key, value = line.strip().split(' = ',1)[1].split(': ',1)
            samples[-1][key] = value
    metadata = pd.DataFrame(samples)
    metadata['sample_id'] = [treatments[x['stress treatment']]+'_'+x['biological replicate'].replace('Rep','') for x in samples]
    assert set(metadata.sample_id) == set(counts.columns)
    metadata.to_csv(OUT/'samples.tsv', sep='\t', index=False)
    runs = pd.read_csv(BASE/'raw/geo/GSE146206/SRA_RunInfo.csv')
    joined = runs.merge(metadata, left_on='SampleName', right_on='gsm', validate='many_to_one')
    assert len(joined) == len(runs)
    joined.to_csv(OUT/'run_sample_mapping.tsv', sep='\t', index=False)
    report = {'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'targets':len(df), 'genes':df.AGI.nunique(), 'samples':len(metadata),
              'runs':len(runs), 'runinfo_size_MB':float(runs.size_MB.sum()),
              'tpm_column_sums':tpm.sum().to_dict(),
              'note':'Author kallisto estimated counts are fractional; retain values. Mannitol is osmotic stress, not drought. Differential results are outcomes, not input features.'}
    (OUT/'extraction_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='tpm_column_sums'}, indent=2))

if __name__ == '__main__':
    main()
