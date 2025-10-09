import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

class EdgeIndexCache: # save sapce, not important
    def __init__(self, device='cuda'):
        self.cache = {}
        self.device = device
        
    def get_edge_index(self, num_nodes, batch_size):
        key = (num_nodes, batch_size)
        if key not in self.cache:
            row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
            edge_index_single = torch.stack([row.flatten(), col.flatten()], dim=0)
            edge_indices = []
            for b in range(batch_size):
                edge_indices.append(edge_index_single + b * num_nodes)
            self.cache[key] = torch.cat(edge_indices, dim=1).to(self.device)
        return self.cache[key]

# !!! Motivation : Use multiheadattetion, not pooling!
class MultiModalGraphEncoder(nn.Module):
    def __init__(self, clip_dim, center_dim, extent_dim, edge_dim, hidden_dim=64, out_dim=128, heads=4):
        super().__init__()
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        self.center_proj = nn.Linear(center_dim, hidden_dim)
        self.extent_proj = nn.Linear(extent_dim, hidden_dim)        
        self.gat1 = GATv2Conv(hidden_dim * 3, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_dim, concat=False)        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, clip_feats, center_feats, extent_feats, edge_index, edge_attr, batch):
        clip_proj = torch.relu(self.clip_proj(clip_feats))
        center_proj = torch.relu(self.center_proj(center_feats))
        extent_proj = torch.relu(self.extent_proj(extent_feats))        
        x = torch.cat([clip_proj, center_proj, extent_proj], dim=-1)        
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        
        # Graph-level attention
        unique_batches = torch.unique(batch)
        graph_embs = []
        for b in unique_batches:
            mask = (batch == b)
            node_embs = x[mask].unsqueeze(0)
            attn_output, _ = self.attention(node_embs, node_embs, node_embs)
            graph_emb = attn_output.mean(dim=1)
            graph_embs.append(graph_emb)
        
        graph_emb = torch.cat(graph_embs, dim=0)
        return self.mlp_out(graph_emb)

class CustomActor(GaussianMixin, Model): # keep the name same with previous version
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        
        self.device = device
        self.img_dim = observation_space["img"].shape[0]
        self.num_nodes = observation_space["graph"]["node_clip"].shape[0]
        
        self.edge_cache = EdgeIndexCache(device=device)
        self.rel_embed = nn.Embedding(26, kwargs.get('edge_dim', 32))
        
        self.graph_encoder = MultiModalGraphEncoder(
            clip_dim=observation_space["graph"]["node_clip"].shape[1],
            center_dim=observation_space["graph"]["node_center"].shape[1],
            extent_dim=observation_space["graph"]["node_extent"].shape[1],
            edge_dim=kwargs.get('edge_dim', 32),
            hidden_dim=kwargs.get('gnn_hidden', 64),
            out_dim=kwargs.get('gnn_out', 128),
            heads=kwargs.get('gnn_heads', 4)
        )
        
        # !!! The motivation here is to deal with visual + scene graph independly
        # For Image  
        self.visual_net = nn.Sequential(
            nn.Linear(self.img_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )        
        # For Scene Graph
        self.graph_net = nn.Sequential(
            nn.Linear(kwargs.get('gnn_out', 128), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )        
        # Fusion and action output
        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_actions),
            nn.Tanh()
        )        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role=""): # Handle different input separately !  No pooling
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        
        img = states["img"].to(self.device)
        clip_feats = states["graph"]["node_clip"].reshape(B * self.num_nodes, -1).to(self.device)
        center_feats = states["graph"]["node_center"].reshape(B * self.num_nodes, -1).to(self.device)
        extent_feats = states["graph"]["node_extent"].reshape(B * self.num_nodes, -1).to(self.device)
        
        rel_ids = states["graph"]["rel_ids"].long().clamp(0, 25).to(self.device)
        edge_attr = self.rel_embed(rel_ids.reshape(B * self.num_nodes * self.num_nodes))
        
        batch = torch.repeat_interleave(torch.arange(B, device=self.device), self.num_nodes)
        edge_index = self.edge_cache.get_edge_index(self.num_nodes, B)
        
        graph_emb = self.graph_encoder(clip_feats, center_feats, extent_feats, edge_index, edge_attr, batch)
        visual_emb = self.visual_net(img)
        graph_emb_processed = self.graph_net(graph_emb)
        
        fused = torch.cat([visual_emb, graph_emb_processed], dim=-1)
        mu = self.fusion_net(fused)
        
        return mu, self.log_std_parameter, {}

class CustomCritic(DeterministicMixin, Model): # keep the name same with previous version
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, kwargs.get('clip_actions', False))
        
        self.device = device
        self.img_dim = observation_space["img"].shape[0]
        self.num_nodes = observation_space["graph"]["node_clip"].shape[0]
        
        self.edge_cache = EdgeIndexCache(device=device)
        self.rel_embed = nn.Embedding(26, kwargs.get('edge_dim', 32))
        
        self.graph_encoder = MultiModalGraphEncoder(
            clip_dim=observation_space["graph"]["node_clip"].shape[1],
            center_dim=observation_space["graph"]["node_center"].shape[1],
            extent_dim=observation_space["graph"]["node_extent"].shape[1],
            edge_dim=kwargs.get('edge_dim', 32),
            hidden_dim=kwargs.get('gnn_hidden', 64),
            out_dim=kwargs.get('gnn_out', 128),
            heads=kwargs.get('gnn_heads', 4)
        )
        
        # !!! The motivation here is to deal with visual + scene graph independly

        # For Image 
        self.visual_net = nn.Sequential(
            nn.Linear(self.img_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )        
        # For Scene Graph
        self.graph_net = nn.Sequential(
            nn.Linear(kwargs.get('gnn_out', 128), 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )        
        # Action 
        self.action_net = nn.Sequential(
            nn.Linear(self.num_actions, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )        
        # Q-value prediction
        self.q_net = nn.Sequential(
            nn.Linear(64 + 32 + 32, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role=""):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        
        img = states["img"].to(self.device)
        clip_feats = states["graph"]["node_clip"].reshape(B * self.num_nodes, -1).to(self.device)
        center_feats = states["graph"]["node_center"].reshape(B * self.num_nodes, -1).to(self.device)
        extent_feats = states["graph"]["node_extent"].reshape(B * self.num_nodes, -1).to(self.device)
        actions = inputs["taken_actions"].to(self.device)
        
        rel_ids = states["graph"]["rel_ids"].long().clamp(0, 25).to(self.device)
        edge_attr = self.rel_embed(rel_ids.reshape(B * self.num_nodes * self.num_nodes))
        
        batch = torch.repeat_interleave(torch.arange(B, device=self.device), self.num_nodes)
        edge_index = self.edge_cache.get_edge_index(self.num_nodes, B)

        # Handle different input separately ! 
        graph_emb = self.graph_encoder(clip_feats, center_feats, extent_feats, edge_index, edge_attr, batch)
        visual_emb = self.visual_net(img)
        graph_emb_processed = self.graph_net(graph_emb)
        action_emb = self.action_net(actions)
        
        fused = torch.cat([visual_emb, graph_emb_processed, action_emb], dim=-1)
        q_value = self.q_net(fused)
        
        return q_value, {}