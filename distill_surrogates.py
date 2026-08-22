import argparse, os, time
import torch

from model import TinyDiT, BlockSpec, DiTBlock
from diffusion import get_teacher_hidden_states
from data import get_dataloaders


def load_teacher(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    dim, heads, depth = ckpt["dim"], ckpt["heads"], ckpt["depth"]
    specs = [BlockSpec("original", dim, dim // 2, 4.0, 2.0, heads) for _ in range(depth)]
    model = TinyDiT(dim=dim, num_heads=heads, block_specs=specs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dim, heads, depth


def train_surrogate_block(student, teacher, get_input_target_fn, steps, lr, device, tag=""):
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        h_in, target, c = get_input_target_fn()
        pred = student(h_in, c)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps:
            print(f"  [{tag}] step {step}/{steps} feature-MSE={loss.item():.5f} "
                  f"({time.time()-t0:.0f}s elapsed)")
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher/teacher_ckpt.pt")
    ap.add_argument("--steps_per_surrogate", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--out_path", type=str, default="./runs/surrogates.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    teacher, dim, heads, depth = load_teacher(args.teacher_ckpt, device)
    print(f"Loaded teacher: dim={dim} heads={heads} depth={depth}")
    train_loader, _ = get_dataloaders(batch_size=args.batch_size)
    data_iter = iter(train_loader)

    def fresh_calib_batch():
        nonlocal data_iter
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
        hs, c = get_teacher_hidden_states(teacher, x, t, y, train=False)
        return hs, c

    surrogates = {"dim": dim, "heads": heads, "depth": depth, "blocks": {}}

    n_pairs = depth // 2
    for p in range(n_pairs):
        i = 2 * p
        student = DiTBlock(BlockSpec("original", dim, dim // 2, 4.0, 2.0, heads), cond_dim=dim).to(device)

        def get_io(i=i):
            hs, c = fresh_calib_batch()
            return hs[i], hs[i + 2], c

        train_surrogate_block(student, teacher, get_io, args.steps_per_surrogate, args.lr, device,
                               tag=f"merge_pair{p} (layers {i},{i+1} -> 1)")
        surrogates["blocks"][f"merge_pair{p}"] = {
            "kind": "merge", "covers": [i, i + 1], "state_dict": student.state_dict()
        }

    for i in range(depth):
        for kind, mlp_r, hidr in [("mlp_mod", 2.0, False), ("hid_red", 4.0, True)]:
            spec = BlockSpec(kind, dim, dim // 2, 4.0, 2.0, heads)
            student = DiTBlock(spec, cond_dim=dim).to(device)

            def get_io(i=i):
                hs, c = fresh_calib_batch()
                return hs[i], hs[i + 1], c

            train_surrogate_block(student, teacher, get_io, args.steps_per_surrogate, args.lr, device,
                                   tag=f"{kind}_layer{i}")
            surrogates["blocks"][f"{kind}_layer{i}"] = {
                "kind": kind, "covers": [i], "state_dict": student.state_dict()
            }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(surrogates, args.out_path)
    print(f"Saved {len(surrogates['blocks'])} surrogate blocks to {args.out_path}")


if __name__ == "__main__":
    main()
