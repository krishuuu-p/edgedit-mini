"""
Tiny DiT for FashionMNIST, built so each transformer layer can independently be:
  - "original":  full attention dim (d), full MLP ratio (r_full)
  - "mlp_mod":   full attention dim (d), reduced MLP ratio (r_reduced)
  - "hid_red":   reduced attention dim (d1 < d), full MLP ratio (r_full)

This mirrors EdgeDiT Figure 2 exactly, just at toy scale (d ~ 128 instead of 1152).
"""
import math
import torch
import torch.nn as nn

class BlockSpec:
    """Describes one transformer layer's variant.

    kind: 'original' | 'mlp_mod' | 'hid_red'
    """
    def __init__(self, kind: str, dim: int, dim_low: int, mlp_ratio_full: float, mlp_ratio_low: float, num_heads: int):
        assert kind in ("original", "mlp_mod", "hid_red")
        self.kind = kind
        self.dim = dim
        if kind == "hid_red":
            self.attn_dim = dim_low
        else:
            self.attn_dim = dim
        if kind == "mlp_mod":
            self.mlp_ratio = mlp_ratio_low
        else:
            self.mlp_ratio = mlp_ratio_full
        self.num_heads = num_heads

    def __repr__(self):
        return f"BlockSpec({self.kind}, attn_dim={self.attn_dim}, mlp_ratio={self.mlp_ratio})"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(self, spec: BlockSpec, cond_dim: int):
        super().__init__()
        dim, attn_dim, heads = spec.dim, spec.attn_dim, spec.num_heads
        self.spec = spec
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        self.attn_dim = attn_dim
        self.num_heads = heads if attn_dim % heads == 0 else 1
        self.head_dim = attn_dim // self.num_heads

        hidden = int(dim * spec.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _attn(self, x):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.out_proj(out)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self._attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class TinyDiT(nn.Module):
    """
    img_size: assumed square, divisible by patch_size
    block_specs: list of BlockSpec, one per layer (defines the architecture "configuration vector" a)
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1, dim=128, num_heads=4,
                 num_classes=10, block_specs=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        self.grid = img_size // patch_size

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = LabelEmbedder(num_classes, dim)

        if block_specs is None:
            block_specs = [BlockSpec("original", dim, dim // 2, 4.0, 2.0, num_heads) for _ in range(6)]
        self.blocks = nn.ModuleList([DiTBlock(spec, cond_dim=dim) for spec in block_specs])
        self.final_layer = FinalLayer(dim, patch_size, in_channels, cond_dim=dim)

    def unpatchify(self, x):
        B = x.shape[0]
        p, c, g = self.patch_size, self.in_channels, self.grid
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, x, t, y, train=True, return_features=False):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, train)
        feats = []
        for blk in self.blocks:
            x = blk(x, c)
            if return_features:
                feats.append(x)
        out = self.final_layer(x, c)
        out = self.unpatchify(out)
        if return_features:
            return out, feats
        return out
