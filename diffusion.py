import torch
import torch.nn as nn


class GaussianDiffusion:
    """Minimal DDPM (Ho et al.) - linear beta schedule."""
    def __init__(self, timesteps=1000, device="cpu"):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ac = torch.sqrt(1.0 - self.alphas_cumprod)
        self.betas = betas
        self.alphas = alphas
        self.device = device

    def q_sample(self, x0, t, noise):
        sac = self.sqrt_ac[t].view(-1, 1, 1, 1)
        s1m = self.sqrt_1m_ac[t].view(-1, 1, 1, 1)
        return sac * x0 + s1m * noise

    def training_loss(self, model, x0, y, train=True):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = model(xt, t, y, train=train)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, shape, y, device, cfg_scale=1.0, num_classes=10, steps=None):
        """Respaced DDIM sampling over the full training noise schedule.

        ``steps`` controls how many denoising evaluations are used, while always
        starting at timestep T-1.  This is essential for fast previews: starting
        from random noise at timestep 99 when the model was trained up to 999
        produces invalid, noise-like samples.
        """
        steps = steps or self.T
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, {self.T}], got {steps}")
        x = torch.randn(shape, device=device)
        schedule = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for index, step_t in enumerate(schedule):
            t = torch.full((shape[0],), step_t.item(), device=device, dtype=torch.long)
            if cfg_scale != 1.0:
                y_null = torch.full_like(y, num_classes)
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                y_in = torch.cat([y, y_null], dim=0)
                eps = model(x_in, t_in, y_in, train=False)
                eps_cond, eps_uncond = eps.chunk(2, dim=0)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t, y, train=False)
            alpha_bar_t = self.alphas_cumprod[step_t]
            x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            if index + 1 == len(schedule):
                x = x0
            else:
                alpha_bar_prev = self.alphas_cumprod[schedule[index + 1]]
                x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1 - alpha_bar_prev) * eps
        return x.clamp(-1, 1)


@torch.no_grad()
def get_teacher_hidden_states(teacher, x, t, y, train=False):
    """Returns hs[0..L] where hs[i] is the hidden state entering block i
    (hs[L] is the state after the last block), plus the conditioning vector c."""
    h = teacher.patch_embed(x).flatten(2).transpose(1, 2) + teacher.pos_embed
    c = teacher.t_embedder(t) + teacher.y_embedder(y, train)
    hs = [h]
    for blk in teacher.blocks:
        h = blk(h, c)
        hs.append(h)
    return hs, c


class AssembledDiT(nn.Module):
    """Wraps a chosen list of (already trained) DiTBlock modules with the
    teacher's shared patch/pos embed, conditioning embedders, and final layer.
    This IS the "configuration vector" a = (b_1, ..., b_L) from EdgeDiT Sec 3.3."""
    def __init__(self, teacher, block_list, share_head=True):
        super().__init__()
        self.patch_embed = teacher.patch_embed
        self.pos_embed = teacher.pos_embed
        self.t_embedder = teacher.t_embedder
        self.y_embedder = teacher.y_embedder
        self.blocks = nn.ModuleList(block_list)
        self.final_layer = teacher.final_layer
        self.img_size = teacher.img_size
        self.patch_size = teacher.patch_size
        self.in_channels = teacher.in_channels
        self.grid = teacher.grid

    def unpatchify(self, x):
        B = x.shape[0]
        p, c, g = self.patch_size, self.in_channels, self.grid
        x = x.reshape(B, g, g, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, x, t, y, train=True):
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, train)
        for blk in self.blocks:
            h = blk(h, c)
        out = self.final_layer(h, c)
        return self.unpatchify(out)
