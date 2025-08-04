import torch
import numpy as np
from torch.nn import init
import torch.nn as nn
from models.HiVit import HiViT_base

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid
    
class Visual_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()        
        self.resize_to = args.resize_to
        self.token_num = args.token_num
        self.encoder = args.encoder
        self.model = self.load_model(args)
        # self.writer = writer

    def load_model(self, args):
        assert args.resize_to[0] % args.patch_size == 0
        assert args.resize_to[1] % args.patch_size == 0
        model = HiViT_base(40)
        return model
    
    # @torch.no_grad()
    def forward(self, frames, step=0):
        # :arg frames:  (B, 3, H, W)
        # :return x:  (B, token, 768)
        B = frames.shape[0]
        x = self.model(frames)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # === Token calculations ===
        slot_dim = args.slot_dim
        hidden_dim = 2048

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 512 + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, slot_maps):
        # :arg slot_maps: (B * S, token, D_slot)

        slot_maps = self.relu(self.layer1(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer2(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer3(slot_maps))    # (B * S, token, D_hidden)

        slot_maps = self.layer4(slot_maps)               # (B * S, token, 768 + 1)

        return slot_maps

class SelfAttentionRefine(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.scale = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        qkv = self.qkv_proj(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        out = attn_probs @ v
        out = self.out_proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class MLSA(nn.Module):
    """
    Slot Attention with *cross-modal* K / V :
        • Keys  ← high-level  (DINO)
        • Values ← low-level  (WSN)
        • Queries / Slots ← original way
    """
    def __init__(self, args,
                 d_dino: int = 512,
                 d_wsn : int = 64):
        super().__init__()
        self.num_slots = args.num_slots
        self.scale     = args.slot_dim ** -0.5
        self.iters     = args.slot_att_iter
        self.slot_dim  = args.slot_dim
        self.query_opt = args.query_opt

        # ---------- Slot initialization ----------
        if self.query_opt:
            self.slots = nn.Parameter(torch.empty(1, self.num_slots, self.slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu  = nn.Parameter(torch.empty(1, 1, self.slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        # ---------- Slot-Attention  ----------
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm         = nn.LayerNorm(self.slot_dim)
        self.update_norm  = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4 * self.slot_dim,
                       self.slot_dim, residual=True, layer_order="pre")

        # ---------- Cross-modal K / V ----------
        #   K : DINO → d_slot
        #   V : WSN  → d_slot
        self.K_dino = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim, bias=False))
        self.K_wsn  = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim,  self.slot_dim, bias=False))
        self.V_dino  = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim,  self.slot_dim, bias=False))
        
        self.init_dino = nn.Sequential(
            nn.LayerNorm(d_dino),
            nn.Linear(d_dino, d_dino),
            nn.ReLU(inplace=True),
            nn.Linear(d_dino, self.slot_dim),
            nn.LayerNorm(self.slot_dim)
        )
        self.init_wsn  = nn.Sequential(
            nn.LayerNorm(d_wsn),
            nn.Linear(d_wsn, d_wsn),
            nn.ReLU(inplace=True),
            nn.Linear(d_wsn,  self.slot_dim),
            nn.LayerNorm(self.slot_dim)
        )

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)
        self.beta = 1

        self.gate_mlp = nn.Sequential(nn.LayerNorm(self.slot_dim),
                                      nn.Linear(self.slot_dim, self.slot_dim),
                                      nn.Sigmoid())   # 0 ~ 1

    # ------------------------------------------------------------
    def forward(self,
                dino_tok: torch.Tensor,   # (B, N, d_dino)
                wsn_tok : torch.Tensor):  # (B, N, d_wsn)
        """
        returns:
            slots      : (B, S, D_slot)
            attn_stack : list[(B, S, N)], len = iters
        """
        B, N, _ = dino_tok.shape
        eps = 1e-8
        S  = self.num_slots
        D  = self.slot_dim

        # ----- Slot initialization -----
        if self.query_opt:
            slots = self.slots.expand(B, S, D)           # learned query slots
        else:
            mu, logsigma = self.slots_mu, self.slots_logsigma
            mu     = mu.expand(B, S, D)
            sigma  = logsigma.exp().expand(B, S, D)
            slots  = mu + sigma * torch.randn_like(mu)

        
        slots = self.norm(slots)
        queries = self.Q(slots)                        # (B,S,D)

        # ----- Modal-specific initial MLP -----
        dino_emb = self.init_dino(dino_tok)                # (B,N,D_slot)
        wsn_emb  = self.init_wsn (wsn_tok )                # (B,N,D_slot)

        # ----- Keys / Values -----
        keys   = self.K_dino(dino_emb)                     # (B,N,D_slot)   
        values = self.V_dino(dino_emb)                     # (B,N,D_slot)
        
        attn_list = []
                

        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm(slots)
            queries = self.Q(slots)                  # (B,S,D)

            dots  = torch.einsum('bsd,bnd->bsn', queries, keys) * self.scale
            attn  = dots.softmax(dim=1) + eps
            attn  = attn / attn.sum(dim=-1, keepdim=True)        # (B,S,N)

            # ── gate is calculated only in the first iteration (t==0) ──
            if t == 0:
                ms   = 1. + (attn.detach() - attn.detach().mean(dim=-1, keepdim=True))
                Mk   = wsn_emb.unsqueeze(1) * ms.unsqueeze(3)                              # (B, N, D)

            # WSN·DINO Hadamard product and slot aggregation
            updates = torch.einsum('bsn,bsnd->bsd', attn, Mk * values.unsqueeze(1))

            slots_gru = self.gru(updates.reshape(-1, D),
                                slots_prev.reshape(-1, D)).view(B, S, D)           
            slots = self.mlp(slots_gru)
            attn_list.append(attn)
            
        return slots, attn, Mk, attn_list
                
class SlotSAR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.slot_dim = args.slot_dim
        self.slot_num = args.num_slots
        self.token_num = args.token_num

        self.mlsa = MLSA(args)
        self.slot_decoder = Decoder(args)

        self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec, mean=0., std=.02)
        
        self.norm_sc = nn.BatchNorm2d(1)
        self.self_attn1 = SelfAttentionRefine(dim=64)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )      

    def sbd_slots(self, slots):
        # :arg slots: (B, S, D_slot)
        # 
        # :return slots: (B, S, token, D_slot)
        B, S, D_slot = slots.shape

        slots = slots.view(-1, 1, D_slot)                   # (B * S, 1, D_slot)
        slots = slots.tile(1, self.token_num, 1)            # (B * S, token, D_slot)

        pos_embed = self.pos_dec.expand(slots.shape)
        slots = slots + pos_embed                          # (B * S, token, D_slot)
        slots = slots.view(B, S, self.token_num, D_slot)

        return slots
    
    
    def reconstruct_feature_map(self, slot_maps):
        # :arg slot_maps: (B, S, token, 768 + 1)
        #
        # :return reconstruction: (B, token, 768)
        # :return masks: (B, S, token)

        B = slot_maps.shape[0]

        channels, masks = torch.split(slot_maps, [512, 1], dim=-1)  # (B, S, token, 768), (B, S, token, 1)
        masks = masks.softmax(dim=1)                                # (B, S, token, 1)

        reconstruction = torch.sum(channels * masks, dim=1)         # (B, token, 768)
        masks = masks.squeeze(dim=-1)                               # (B, S, token)

        return reconstruction, masks


    def forward(self, vis_features, sc_features):   
        B, C, H, W = sc_features.shape        
        C = 64        
        sc_features = self.norm_sc(sc_features)       
        sc_features = self.block1(sc_features)
        sc_features = self.self_attn1(sc_features)        
        sc_features = sc_features.view(B, C, H * W).permute(0, 2, 1)  # (B, 1024, 64)
        
        slots, attn, Mk, attn_list = self.mlsa(vis_features, sc_features)                           # (B, S, D_slot), (B, S, token)

        slot_maps = self.sbd_slots(slots)
        slot_maps = self.slot_decoder(slot_maps)                            # (B, S, token, 768 + 1)

        reconstruction, masks = self.reconstruct_feature_map(slot_maps)     # (B, token, 768), (B, S, token)

        return reconstruction, slots, masks, attn, Mk, attn_list