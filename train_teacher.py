import argparse, os, time, json
import torch
from torchvision.utils import save_image

from model import TinyDiT, BlockSpec
from diffusion import GaussianDiffusion
from data import get_dataloaders


def build_teacher(dim, num_heads, depth, mlp_ratio_full=4.0):
    specs = [BlockSpec("original", dim, dim // 2, mlp_ratio_full, mlp_ratio_full / 2, num_heads) for _ in range(depth)]
    return TinyDiT(img_size=28, patch_size=4, in_channels=1, dim=dim, num_heads=num_heads,
                    num_classes=10, block_specs=specs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--out_dir", type=str, default="./runs/teacher")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--sample_steps", type=int, default=100)  # DDPM steps used for preview samples (fast, lower quality)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    print(f"Device: {device}")

    train_loader, _ = get_dataloaders(batch_size=args.batch_size)
    model = build_teacher(args.dim, args.heads, args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {n_params/1e6:.2f}M, depth={args.depth}, dim={args.dim}")

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_log = []
    data_iter = iter(train_loader)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        loss = diffusion.training_loss(model, x, y, train=True)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step / elapsed
            eta_min = (args.steps - step) / sps / 60
            print(f"step {step}/{args.steps} loss={loss.item():.4f} "
                  f"({sps:.2f} steps/s, ETA {eta_min:.1f} min)")
            loss_log.append({"step": step, "loss": loss.item()})

        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            y_sample = torch.arange(10, device=device) % 10
            with torch.no_grad():
                samples = diffusion.sample(model, (10, 1, 28, 28), y_sample, device,
                                            steps=args.sample_steps)
            save_image((samples + 1) / 2, os.path.join(args.out_dir, f"sample_step{step}.png"), nrow=10)
            torch.save({"model": model.state_dict(), "step": step,
                        "dim": args.dim, "heads": args.heads, "depth": args.depth},
                       os.path.join(args.out_dir, "teacher_ckpt.pt"))
            model.train()

    with open(os.path.join(args.out_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Done. Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
