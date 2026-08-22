import argparse, json, csv, os
import torch
from complexity import estimate_macs

from diffusion import GaussianDiffusion
from distill_surrogates import load_teacher
from search import assemble_from_names, measure_latency
from data import get_dataloaders
from proxy_fid import SmallCNN, train_classifier, proxy_fid


@torch.no_grad()
def generate_samples(model, diffusion, n, device, sample_steps):
    y = torch.randint(0, 10, (n,), device=device)
    return diffusion.sample(model, (n, 1, 28, 28), y, device, steps=sample_steps)


def collect_real_images(loader, n, device):
    """Collect exactly n test images instead of being limited to one loader batch."""
    batches = []
    total = 0
    for images, _ in loader:
        take = min(n - total, images.shape[0])
        batches.append(images[:take])
        total += take
        if total == n:
            return torch.cat(batches, dim=0).to(device)
    raise RuntimeError(f"Test loader contained only {total} images; needed {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher/teacher_ckpt.pt")
    ap.add_argument("--surrogates", type=str, default="./runs/surrogates.pt")
    ap.add_argument("--search_results", type=str, default="./runs/search/search_results.json")
    ap.add_argument("--finetuned_dir", type=str, default="./runs/finetuned")
    ap.add_argument("--out_dir", type=str, default="./runs/benchmark")
    ap.add_argument("--n_fid_samples", type=int, default=200)
    ap.add_argument("--sample_steps", type=int, default=250)
    ap.add_argument("--classifier_epochs", type=int, default=3)
    ap.add_argument("--metric_device", choices=["cpu", "cuda"], default="cpu",
                    help="device for the small Proxy-FID classifier; CPU is safer on memory-limited GPUs")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    metric_device = args.metric_device
    if metric_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--metric_device cuda was requested, but CUDA is unavailable")
    os.makedirs(args.out_dir, exist_ok=True)

    teacher, dim, heads, depth = load_teacher(args.teacher_ckpt, device)
    surrogates = torch.load(args.surrogates, map_location=device)
    for b in surrogates["blocks"].values():
        for k in b["state_dict"]:
            b["state_dict"][k] = b["state_dict"][k].to(device)
    with open(args.search_results) as f:
        sr = json.load(f)

    # The metric classifier is deliberately kept separate from diffusion-model
    # inference. num_workers=0 prevents Windows worker processes from causing
    # host-memory pressure after a GPU-heavy search stage.
    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    print(f"Training small classifier for proxy-FID features on {metric_device}...")
    classifier = train_classifier(train_loader, metric_device, epochs=args.classifier_epochs)

    # Real reference set for Proxy-FID; match the requested generated sample count.
    real_x = collect_real_images(test_loader, args.n_fid_samples, device)

    models = {}
    models["DiT-teacher (baseline)"] = (teacher, sum(p.numel() for p in teacher.parameters()))

    for tag, key in [("EdgeDiT-small", "selected_smallest"), ("EdgeDiT-large", "selected_largest")]:
        names = sr[key]["layout"]
        model = assemble_from_names(names, teacher, surrogates, dim, heads, device)
        ckpt_path = os.path.join(args.finetuned_dir, tag, "ckpt.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded fine-tuned weights for {tag}")
        else:
            print(f"WARNING: no fine-tuned checkpoint found for {tag} at {ckpt_path}; "
                  f"using distilled-but-not-finetuned weights.")
        model.eval()
        models[tag] = (model, sum(p.numel() for p in model.parameters()))

    rows = []
    dummy = torch.randn(1, 1, 28, 28, device=device)
    dummy_t = torch.zeros(1, dtype=torch.long, device=device)
    dummy_y = torch.zeros(1, dtype=torch.long, device=device)
    teacher_params = models["DiT-teacher (baseline)"][1]

    for name, (model, params) in models.items():
        macs = estimate_macs(model)
        gflops = macs * 2 / 1e9
        gmacs = macs / 1e9
        per_step_ms = measure_latency(model, device)
        est_total_latency_ms = per_step_ms * args.sample_steps

        print(f"Generating {args.n_fid_samples} samples for {name} ({args.sample_steps} steps)...")
        fakes = generate_samples(model, diffusion, args.n_fid_samples, device, args.sample_steps)
        fid = proxy_fid(classifier, real_x, fakes, metric_device)

        rows.append({
            "Model": name,
            "Params (M)": round(params / 1e6, 3),
            "Params vs teacher": f"{100*(1 - params/teacher_params):.1f}%" if name != "DiT-teacher (baseline)" else "-",
            "GMACs": round(gmacs, 4),
            "GFLOPs": round(gflops, 4),
            "Per-step latency (ms, your GPU)": round(per_step_ms, 3),
            f"Est. {args.sample_steps}-step generation latency (ms)": round(est_total_latency_ms, 1),
            "Proxy-FID (FashionMNIST-CNN feats)": round(fid, 3),
        })
        print(rows[-1])

    csv_path = os.path.join(args.out_dir, "benchmark_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out_dir, "benchmark_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved benchmark table to {csv_path}")
    print("\n(Reminder: latency is measured on YOUR GPU as a stand-in for the paper's mobile")
    print("NPU numbers - report it explicitly as a proxy, not a direct comparison to Table 3.)")


if __name__ == "__main__":
    main()
