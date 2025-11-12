import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean, scatter_max

from models.graph_encoder import GraphEncoder, LevelWisePooling
from models.recipe_encoder import RecipeEncoder, PositionalEncoding
from models.transformer_decoder import TransformerDecoder, generate_causal_mask


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, max_hop=3):
        super(MultiScaleFeatureFusion, self).__init__()
        self.max_hop = max_hop

        self.hop_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max_hop)
        ])

        self.scale_gate = nn.Sequential(
            nn.Linear(2, hidden_dim // 4), 
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, max_hop),
            nn.Sigmoid()
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * max_hop, max_hop),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Linear(hidden_dim * max_hop, hidden_dim)

    def forward(self, node_embeddings, edge_index, node_depths=None):
        hop_features = []
        current_feat = node_embeddings

        num_nodes = node_embeddings.size(0)
        max_depth = node_depths.max().float() if node_depths is not None else torch.tensor(10.0).to(
            node_embeddings.device)
        scale_features = torch.tensor([
            num_nodes / 5000.0,  
            max_depth / 20.0  
        ], device=node_embeddings.device).unsqueeze(0)

        scale_weights = self.scale_gate(scale_features) 

        for hop in range(self.max_hop):
            transformed = self.hop_transforms[hop](current_feat)
            transformed = transformed * scale_weights[0, hop]
            hop_features.append(transformed)

            if hop < self.max_hop - 1:
                row, col = edge_index
                current_feat = scatter_mean(current_feat[row], col,
                                            dim=0, dim_size=node_embeddings.size(0))
        concat_features = torch.cat(hop_features, dim=-1)
        attention_weights = self.attention(concat_features)

        weighted_features = []
        for i in range(self.max_hop):
            weighted_features.append(hop_features[i] * attention_weights[:, i:i + 1])

        fused = torch.cat(weighted_features, dim=-1)
        return self.fusion(fused)


class DynamicGraphSequenceInteraction(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicGraphSequenceInteraction, self).__init__()

        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim))

        self.graph_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())

        self.graph_modulator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh())

        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid())

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, graph_seq, recipe_emb):
        seq_len, batch_size, hidden_dim = recipe_emb.shape
        graph_len, _, graph_dim = graph_seq.shape

        graph_seq_expanded = graph_seq.expand(-1, batch_size, -1)

        graph_global = torch.mean(graph_seq_expanded, dim=0, keepdim=True)
        graph_global = self.graph_aggregator(graph_global)
        graph_global = graph_global.expand(seq_len, -1, -1)

        step_context = self.step_encoder(recipe_emb)

        graph_modulated = self.graph_modulator(torch.cat([graph_global, step_context], dim=-1))

        gate_input = torch.cat([recipe_emb, graph_modulated, step_context], dim=-1)
        gate = self.adaptive_gate(gate_input)

        fused = torch.cat([recipe_emb, graph_modulated * gate], dim=-1)
        output = self.fusion(fused)

        return output


class CircScale(nn.Module):
    def __init__(self, input_node_dim, hidden_dim, num_heuristics,
                 nhead=8, dim_feedforward=2048, dropout=0.3, num_layers=1,
                 task_type="delay",
                 use_multiscale=False, max_hop=3,
                 use_dynamic_interaction=False):
        super(CircScale, self).__init__()

        self.task_type = task_type
        self.use_dynamic_interaction = use_dynamic_interaction
        self.graph_encoder = GraphEncoder(input_node_dim, hidden_dim)

        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_fusion = MultiScaleFeatureFusion(hidden_dim, max_hop)

        self.level_pooling = LevelWisePooling(hidden_dim)

        self.recipe_encoder = RecipeEncoder(num_heuristics, 2 * hidden_dim)
        self.pos_encoder = PositionalEncoding(2 * hidden_dim, dropout)

        if use_dynamic_interaction:
            self.dynamic_interaction = DynamicGraphSequenceInteraction(2 * hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerDecoder(2 * hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.qor_head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, batch):
        graph_batch = batch['graph']
        recipe_indices = batch['recipe_indices']
        node_depths = batch['node_depths']
        device = recipe_indices.device

        node_features, edge_index = graph_batch.x, graph_batch.edge_index
        node_embeddings = self.graph_encoder(node_features, edge_index)

        if self.use_multiscale:
            node_embeddings = self.multiscale_fusion(node_embeddings, edge_index, node_depths)

        graph_seq = self.level_pooling(node_embeddings, node_depths)
        recipe_embeddings = self.recipe_encoder(recipe_indices)
        recipe_embeddings = self.pos_encoder(recipe_embeddings)

        seq_len = recipe_indices.size(1)
        causal_mask = generate_causal_mask(seq_len, device)

        recipe_embeddings = recipe_embeddings.transpose(0, 1)
        batch_size = recipe_indices.size(0)
        graph_seq = graph_seq.unsqueeze(1).expand(-1, batch_size, -1)

        if self.use_dynamic_interaction:
            recipe_embeddings = self.dynamic_interaction(graph_seq, recipe_embeddings)

        output = recipe_embeddings
        for layer in self.transformer_layers:
            output = layer(output, graph_seq, causal_mask)

        qor_trajectory = self.qor_head(output).squeeze(-1).transpose(0, 1)

        return qor_trajectory

    def to(self, device):
        model = super().to(device)
        return model
