"""Lightweight, dependency-free MAC estimates for TinyDiT models.

The estimate counts matrix multiplications, convolutions, and attention products
for a single denoising forward pass. Activations, normalisation, and elementwise
operations are intentionally omitted, as is conventional for GMAC reporting.
"""


def estimate_macs(model):
    """Return estimated multiply-accumulates for one image forward pass."""
    dim = model.pos_embed.shape[-1]
    tokens = model.pos_embed.shape[1]
    patch = model.patch_size
    channels = model.in_channels
    grid = model.grid

    macs = grid * grid * dim * channels * patch * patch
    macs += 256 * dim + dim * dim

    for block in model.blocks:
        attn_dim = block.attn_dim
        mlp_hidden = block.mlp[0].out_features
        macs += 3 * tokens * dim * attn_dim
        macs += 2 * tokens * tokens * attn_dim
        macs += tokens * attn_dim * dim
        macs += 2 * tokens * dim * mlp_hidden
        macs += dim * (6 * dim)

    macs += dim * (2 * dim)
    macs += tokens * dim * (patch * patch * channels)
    return int(macs)
