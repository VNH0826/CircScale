# models/graph_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)  # 第一层GCN
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)  # 第二层GCN

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class LevelWisePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(LevelWisePooling, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, node_embeddings, node_depths):
        if node_depths.dim() > 1:
            node_depths = node_depths.squeeze(0)

        max_depth = int(node_depths.max().item()) + 1
        level_embeddings = []

        for level in range(max_depth):
            level_mask = (node_depths == level)
            if level_mask.sum() > 0:
                level_nodes = node_embeddings[level_mask]

                mean_pool = torch.mean(level_nodes, dim=0)  # 平均池化 [hidden_dim]
                max_pool = torch.max(level_nodes, dim=0)[0]  # 最大池化 [hidden_dim]

                level_emb = torch.cat([mean_pool, max_pool], dim=0)  # [2*hidden_dim]
                level_embeddings.append(level_emb)
            else:
                level_embeddings.append(torch.zeros(2 * self.hidden_dim,
                                                    device=node_embeddings.device))
        return torch.stack(level_embeddings)