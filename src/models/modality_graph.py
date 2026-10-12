"""Independent modality-gated graph model; not a claim of architectural novelty.

Inputs are label-free sequence, untreated expression, and domain features.
Normalized sparse PPI adjacency supplies optional neighborhood context.
"""
import torch
from torch import nn


class ModalityGraph(nn.Module):
    def __init__(self, input_dims, hidden=128, tasks=3, dropout=.2, use_graph=True):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
            for d in input_dims
        ])
        self.gate = nn.Linear(hidden, 1)
        self.use_graph = use_graph
        self.layers = nn.ModuleList([nn.Linear(hidden*2, hidden) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, tasks)

    def forward(self, modalities, adjacency=None, available=None):
        encoded = torch.stack([m(x) for m,x in zip(self.encoders,modalities)],dim=1)
        scores = self.gate(encoded).squeeze(-1)
        if available is not None:
            if not available.any(dim=1).all():
                raise ValueError('Each protein needs at least one available modality')
            scores = scores.masked_fill(~available, float('-inf'))
        weights = scores.softmax(dim=1)
        state = (encoded*weights.unsqueeze(-1)).sum(dim=1)
        if self.use_graph:
            if adjacency is None: raise ValueError('Graph requested without adjacency')
            for layer,norm in zip(self.layers,self.norms):
                neighbors = torch.sparse.mm(adjacency,state)
                state = norm(state+self.dropout(torch.nn.functional.gelu(layer(torch.cat([state,neighbors],dim=-1)))))
        return self.head(self.dropout(state)),weights
