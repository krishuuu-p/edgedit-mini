import argparse, json, os, time
import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion, AssembledDiT
from distill_surrogates import load_teacher
from search import assemble_from_names, build_block
from data import get_dataloaders


def assemble_uniform(teacher, surrogates, dim, heads, device, variant):
    """All `depth` layers using the SAME variant ('original' | 'mlp_mod' | 'hid_red')."""
    depth = len(teacher.blocks)
    blocks = []
    for i in range(depth):
        if variant == "original":
            blocks.append(teacher.blocks[i])
        else:
            sd = surrogates["blocks"][f"{variant}_layer{i}"]["state_dict"]
            blocks.append(build_block(variant, dim, heads, sd).to(device))
    return AssembledDiT(teacher, blocks).to(device)


def assemble_merge_k(teacher, surrogates, dim, heads, device, k_pairs):
    """Merge the first k_pairs consecutive pairs, leave the rest as original."""
    depth = len(teacher.blocks)
    n_pairs = depth // 2
    blocks = []
    p = 0
    i = 0
    while i < depth:
        if p < k_pairs and i + 1 < depth:
            sd = surrogates["blocks"][f"merge_pair{p}"]["state_dict"]
            blocks.append(build_block("original", dim, heads, sd).to(device))
            i += 2
        else:
            blocks.append(teacher.blocks[i])
            i += 1
        p += 1
    return AssembledDiT(teacher, blocks).to(device)


@torch.no_grad()
def sample_grid(model, diffusion, device, sample_steps, path, n=10):
    y = torch.arange(n, device=device) % 10
    model.eval()
    s = diffusion.sample(model, (n, 1, 28, 28), y, device, steps=sample_steps)
    save_image((s + 1) / 2, path, nrow=n)


def quick_train(model, diffusion, train_loader, device, steps, lr=1e-4, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    data_iter = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        loss = diffusion.training_loss(model, x, y, train=True)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            print(f"    step {step}/{steps} loss={loss.item():.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher/teacher_ckpt.pt")
    ap.add_argument("--surrogates", type=str, default="./runs/surrogates.pt")
    ap.add_argument("--search_results", type=str, default="./runs/search/search_results.json")
    ap.add_argument("--finetuned_dir", type=str, default="./runs/finetuned")
    ap.add_argument("--out_dir", type=str, default="./runs/figures")
    ap.add_argument("--sample_steps", type=int, default=250)
    ap.add_argument("--no_kd_steps", type=int, default=2000,
                     help="training steps for the random-init 'without FwKD' baseline (Fig 7 analog)")
    ap.add_argument("--skip_no_kd_ablation", action="store_true",
                     help="skip the with/without-FwKD figure to save time")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    teacher, dim, heads, depth = load_teacher(args.teacher_ckpt, device)
    surrogates = torch.load(args.surrogates, map_location=device)
    for b in surrogates["blocks"].values():
        for k in b["state_dict"]:
            b["state_dict"][k] = b["state_dict"][k].to(device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    print("Figure: block-removal severity (merge 1 / 2 / 3 pairs)...")
    for k in range(1, depth // 2 + 1):
        m = assemble_merge_k(teacher, surrogates, dim, heads, device, k)
        sample_grid(m, diffusion, device, args.sample_steps,
                    os.path.join(args.out_dir, f"fig_block_removal_merge{k}pairs.png"))
    sample_grid(teacher, diffusion, device, args.sample_steps,
                os.path.join(args.out_dir, "fig_block_removal_teacher_ref.png"))

    print("Figure: MLP-ratio ablation (all layers r=4 vs r=2)...")
    m_orig = assemble_uniform(teacher, surrogates, dim, heads, device, "original")
    m_mlp = assemble_uniform(teacher, surrogates, dim, heads, device, "mlp_mod")
    sample_grid(m_orig, diffusion, device, args.sample_steps, os.path.join(args.out_dir, "fig_mlp_ratio_r4_original.png"))
    sample_grid(m_mlp, diffusion, device, args.sample_steps, os.path.join(args.out_dir, "fig_mlp_ratio_r2_reduced.png"))

    print("Figure: hidden-dim ablation (d=128 vs d=64)...")
    m_hid = assemble_uniform(teacher, surrogates, dim, heads, device, "hid_red")
    sample_grid(m_hid, diffusion, device, args.sample_steps, os.path.join(args.out_dir, "fig_hidden_dim_d64_reduced.png"))
    if not args.skip_no_kd_ablation:
        print("Figure: with/without FwKD ablation (this involves a short extra training run)...")
        with open(args.search_results) as f:
            sr = json.load(f)
        names = sr["selected_smallest"]["layout"]

        with_kd = assemble_from_names(names, teacher, surrogates, dim, heads, device)
        ckpt_path = os.path.join(args.finetuned_dir, "EdgeDiT-small", "ckpt.pt")
        if os.path.exists(ckpt_path):
            with_kd.load_state_dict(torch.load(ckpt_path, map_location=device))
        sample_grid(with_kd, diffusion, device, args.sample_steps,
                    os.path.join(args.out_dir, "fig_ablation_WITH_fwkd.png"))

        without_kd = assemble_from_names(names, teacher, surrogates, dim, heads, device)
        for p in without_kd.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
        train_loader, _ = get_dataloaders(batch_size=128)
        t0 = time.time()
        quick_train(without_kd, diffusion, train_loader, device, args.no_kd_steps)
        print(f"  random-init baseline trained in {(time.time()-t0)/60:.1f} min")
        sample_grid(without_kd, diffusion, device, args.sample_steps,
                    os.path.join(args.out_dir, "fig_ablation_WITHOUT_fwkd.png"))

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
