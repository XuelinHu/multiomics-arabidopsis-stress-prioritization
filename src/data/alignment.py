"""Explicit protein identifier alignment for label-free inference inputs."""
import numpy as np


def align_protein_arrays(source_ids,values,target_ids):
    source_ids=np.asarray(source_ids);target_ids=np.asarray(target_ids);values=np.asarray(values)
    if len(source_ids)!=len(values):raise ValueError('Protein identifier and feature row counts differ')
    if len(np.unique(source_ids))!=len(source_ids) or len(np.unique(target_ids))!=len(target_ids):
        raise ValueError('Duplicate input protein identifiers')
    lookup={protein:i for i,protein in enumerate(source_ids)}
    if not set(target_ids).issubset(lookup):raise ValueError('Required protein identifier missing')
    aligned=values[[lookup[protein] for protein in target_ids]]
    if not np.isfinite(aligned).all():raise ValueError('Missing/nonfinite numeric features')
    return aligned
