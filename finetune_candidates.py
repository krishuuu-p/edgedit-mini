import argparse, os, json, time
import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from distill_surrogates import load_teacher
from search import assemble_from_names
from data import get_dataloaders


def finetune_one(model, name, args, diffusion, train_loader, device):
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data_iter = iter(train_loader)
    loss_log = []
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
            print(f"  [{name}] step {step}/{args.steps} loss={loss.item():.4f} "
                  f"({sps:.2f} steps/s, ETA {(args.steps-step)/sps/60:.1f} min)")
            loss_log.append({"step": step, "loss": loss.item()})
        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            y_sample = torch.arange(10, device=device) % 10
            with torch.no_grad():
                samples = diffusion.sample(model, (10, 1, 28, 28), y_sample, device,
                                            steps=args.sample_steps)
            save_image((samples + 1) / 2, os.path.join(out_dir, f"sample_step{step}.png"), nrow=10)
            torch.save(model.state_dict(), os.path.join(out_dir, "ckpt.pt"))
            model.train()
    with open(os.path.join(out_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"[{name}] done in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher/teacher_ckpt.pt")
    ap.add_argument("--surrogates", type=str, default="./runs/surrogates.pt")
    ap.add_argument("--search_results", type=str, default="./runs/search/search_results.json")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--sample_every", type=int, default=1500)
    ap.add_argument("--out_dir", type=str, default="./runs/finetuned")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    teacher, dim, heads, depth = load_teacher(args.teacher_ckpt, device)
    surrogates = torch.load(args.surrogates, map_location=device)
    for b in surrogates["blocks"].values():
        for k in b["state_dict"]:
            b["state_dict"][k] = b["state_dict"][k].to(device)

    with open(args.search_results) as f:
        sr = json.load(f)

    train_loader, _ = get_dataloaders(batch_size=args.batch_size)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    for tag, key in [("EdgeDiT-small", "selected_smallest"), ("EdgeDiT-large", "selected_largest")]:
        names = sr[key]["layout"]
        print(f"\n=== Fine-tuning {tag} (depth={len(names)}, params={sr[key]['params']/1e6:.2f}M) ===")
        model = assemble_from_names(names, teacher, surrogates, dim, heads, device)
        finetune_one(model, tag, args, diffusion, train_loader, device)


if __name__ == "__main__":
    main()
