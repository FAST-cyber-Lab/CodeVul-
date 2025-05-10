

import torch
import torch.nn as nn

class DynamicGCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicGCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x, adj_matrix):
        if adj_matrix.shape[1] != x.shape[1]:
            device = x.device
            adj_matrix = torch.eye(x.shape[1], device=device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.linear(x)
        x = torch.matmul(adj_matrix, x) / (torch.sum(adj_matrix, dim=2, keepdim=True) + 1e-6)
        return self.activation(x)

class MultiLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MultiLayerGCN, self).__init__()
        self.gcn1 = DynamicGCNConv(in_features, hidden_features)
        self.gcn2 = DynamicGCNConv(hidden_features, out_features)

    def forward(self, x, adj_matrix):
        x = self.gcn1(x, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        return x

class HybridModel(nn.Module):
    def __init__(self, emb_dim, gcn_hidden, gcn_out):
        super(HybridModel, self).__init__()
        self.gcn = MultiLayerGCN(emb_dim, gcn_hidden, gcn_out)
        self.norm = nn.LayerNorm(gcn_out)

    def forward(self, x, adj_matrix):
        x = self.gcn(x, adj_matrix)
        return self.norm(x)
