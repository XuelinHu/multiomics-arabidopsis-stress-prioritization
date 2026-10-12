"""Two-layer Heterogeneous Graph Transformer (HGT) for stress-response prediction.

Node types: protein (ESM-PCA + control expression), domain (InterPro), go (GO term).
Edge types: (protein, ppi, protein) STRING>=700, (protein, contains, domain),
            (protein, associated, go).
Domain/GO nodes use learnable embeddings; vocab restricted to training-set
annotation classes (same anti-leakage rule as the MLP pipeline).
Full-batch forward; training graph never aggregates validation nodes.
"""
import torch
from torch import nn
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(self, protein_dim, hidden=128, tasks=3, heads=4, num_layers=2, dropout=.2):
        super().__init__()
        self.metadata = (['protein', 'domain', 'go'],
                         [('protein', 'ppi', 'protein'),
                          ('protein', 'contains', 'domain'),
                          ('protein', 'associated', 'go')])
        in_channels = {'protein': protein_dim, 'domain': hidden, 'go': hidden}
        self.domain_emb = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.go_emb = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(HGTConv(hidden, hidden, self.metadata, heads=heads))
            self.norms.append(nn.LayerNorm(hidden))
        self.protein_proj = nn.Sequential(nn.Linear(protein_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Linear(hidden, tasks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, n_domain, n_go):
        x = dict(x_dict)
        x['protein'] = self.protein_proj(x['protein'])
        x['domain'] = self.domain_emb.expand(n_domain, -1)
        x['go'] = self.go_emb.expand(n_go, -1)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index_dict)
            for k in x:
                x[k] = norm(self.dropout(torch.nn.functional.gelu(x[k])))
        return self.head(self.dropout(x['protein']))
