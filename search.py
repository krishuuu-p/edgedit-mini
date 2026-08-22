import argparse, json, os, random, statistics, time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from complexity import estimate_macs

from model import BlockSpec, DiTBlock
from diffusion import AssembledDiT, get_teacher_hidden_states
from distill_surrogates import load_teacher
from data import get_dataloaders

VARIANTS = ["original", "mlp_mod", "hid_red"]


def build_block(kind, dim, heads, state_dict=None):
    spec = BlockSpec(kind, dim, dim // 2, 4.0, 2.0, heads)
    blk = DiTBlock(spec, cond_dim=dim)
    if state_dict is not None:
        blk.load_state_dict(state_dict)
    return blk


def sample_config(n_pairs):
    """Returns a list of per-position choices:
    for each pair, either ('merge', p) or [('keep', i, variant), ('keep', i+1, variant)]"""
    layout = []
    for p in range(n_pairs):
        if random.random() < 0.5:
            layout.append(("merge", p))
        else:
            v1 = random.choice(VARIANTS)
            v2 = random.choice(VARIANTS)
            layout.append(("keep", 2 * p, v1, 2 * p + 1, v2))
    return layout


def assemble_from_layout(layout, teacher, surrogates, dim, heads, device):
    blocks = []
    names = []
    for entry in layout:
        if entry[0] == "merge":
            p = entry[1]
            sd = surrogates["blocks"][f"merge_pair{p}"]["state_dict"]
            blocks.append(build_block("original", dim, heads, sd).to(device))
            names.append(f"merge_pair{p}")
        else:
            _, i1, v1, i2, v2 = entry
            for i, v in [(i1, v1), (i2, v2)]:
                if v == "original":
                    blocks.append(teacher.blocks[i])  # reuse teacher weights directly
                    names.append(f"orig_layer{i}")
                else:
                    sd = surrogates["blocks"][f"{v}_layer{i}"]["state_dict"]
                    blocks.append(build_block(v, dim, heads, sd).to(device))
                    names.append(f"{v}_layer{i}")
    model = AssembledDiT(teacher, blocks).to(device)
    return model, names


@torch.no_grad()
def proxy_quality(model, hs_target, calib_x, calib_t, calib_y, train=False):
    """Cheap quality proxy: MSE between candidate's pre-final-layer hidden state
    and the teacher's, on the same calibration batch. Avoids running full diffusion
    sampling + FID for every candidate during search (mirrors the paper's rationale
    for relaxing the search objective since true FID is too costly to evaluate
    for every candidate)."""
    h = model.patch_embed(calib_x).flatten(2).transpose(1, 2) + model.pos_embed
    c = model.t_embedder(calib_t) + model.y_embedder(calib_y, train)
    for blk in model.blocks:
        h = blk(h, c)
    return torch.nn.functional.mse_loss(h, hs_target).item()


def assemble_from_names(names, teacher, surrogates, dim, heads, device):
    """Rebuild an AssembledDiT from a saved list of block names (as stored in
    search_results.json), e.g. ['merge_pair0', 'orig_layer2', 'mlp_mod_layer3', ...]."""
    blocks = []
    for name in names:
        if name.startswith("orig_layer"):
            i = int(name.split("orig_layer")[1])
            blocks.append(teacher.blocks[i])
        else:
            sd = surrogates["blocks"][name]["state_dict"]
            kind = surrogates["blocks"][name]["kind"]
            build_kind = "original" if kind == "merge" else kind
            blocks.append(build_block(build_kind, dim, heads, sd).to(device))
    return AssembledDiT(teacher, blocks).to(device)


def pareto_front(points):
    """points: list of (params, quality) - lower is better for both. Returns indices on the front."""
    idx = sorted(range(len(points)), key=lambda i: points[i][0])
    front, best_q = [], float("inf")
    for i in idx:
        if points[i][1] < best_q:
            front.append(i)
            best_q = points[i][1]
    return front


@torch.no_grad()
def measure_latency(model, device, warmup=20, iterations=100, repeats=3):
    """Median single-image denoising-step latency in milliseconds.

    CUDA synchronization ensures asynchronous GPU work is included in the
    measurement.  The median of repeated measurements reduces timing noise.
    """
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    t = torch.zeros(1, dtype=torch.long, device=device)
    y = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(warmup):
        model(x, t, y, train=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            model(x, t, y, train=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.median(timings)


def save_pareto_plot(results, front_indices, x_key, x_label, title, path, selected_fast=None, selected_quality=None):
    def x_value(result):
        return result[x_key] / 1e6 if x_key == "params" else result[x_key]

    xs = [x_value(r) for r in results]
    ys = [r["proxy_quality_mse"] for r in results]
    front = [results[i] for i in front_indices]
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, c="lightgray", label="candidates")
    plt.scatter([x_value(r) for r in front], [r["proxy_quality_mse"] for r in front],
                c="tab:blue", label="Pareto front")
    if selected_fast is not None:
        plt.scatter([x_value(selected_fast)], [selected_fast["proxy_quality_mse"]], c="tab:green",
                    marker="*", s=200, label="EdgeDiT-small (fastest)")
    if selected_quality is not None:
        plt.scatter([x_value(selected_quality)], [selected_quality["proxy_quality_mse"]], c="tab:red",
                    marker="*", s=200, label="EdgeDiT-large (quality)")
    plt.xlabel(x_label)
    plt.ylabel("Proxy quality (feature MSE, lower=better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher/teacher_ckpt.pt")
    ap.add_argument("--surrogates", type=str, default="./runs/surrogates.pt")
    ap.add_argument("--num_candidates", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="./runs/search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency_warmup", type=int, default=20)
    ap.add_argument("--latency_iters", type=int, default=100)
    ap.add_argument("--latency_repeats", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    teacher, dim, heads, depth = load_teacher(args.teacher_ckpt, device)
    surrogates = torch.load(args.surrogates, map_location=device)
    for b in surrogates["blocks"].values():
        for k in b["state_dict"]:
            b["state_dict"][k] = b["state_dict"][k].to(device)

    train_loader, _ = get_dataloaders(batch_size=64)
    calib_x, calib_y = next(iter(train_loader))
    calib_x, calib_y = calib_x.to(device), calib_y.to(device)
    calib_t = torch.randint(0, 1000, (calib_x.shape[0],), device=device)
    hs, _ = get_teacher_hidden_states(teacher, calib_x, calib_t, calib_y, train=False)
    hs_target = hs[-1]

    n_pairs = depth // 2
    results = []
    seen_layouts = set()
    attempts = 0
    while len(results) < args.num_candidates and attempts < args.num_candidates * 5:
        attempts += 1
        layout = sample_config(n_pairs)
        key = str(layout)
        if key in seen_layouts:
            continue
        seen_layouts.add(key)

        model, names = assemble_from_layout(layout, teacher, surrogates, dim, heads, device)
        params = sum(p.numel() for p in model.parameters())
        macs = estimate_macs(model)
        quality = proxy_quality(model, hs_target, calib_x, calib_t, calib_y)
        latency_ms = measure_latency(model, device, args.latency_warmup, args.latency_iters, args.latency_repeats)

        results.append({"layout": names, "params": params, "gmacs": macs / 1e9,
                         "proxy_quality_mse": quality, "latency_ms": latency_ms, "depth": len(names)})
        print(f"[{len(results)}/{args.num_candidates}] depth={len(names)} params={params/1e6:.2f}M "
              f"gmacs={macs/1e9:.4f} latency={latency_ms:.3f}ms proxy_mse={quality:.5f}")

    # teacher baseline point
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_macs = estimate_macs(teacher)
    teacher_latency_ms = measure_latency(teacher, device, args.latency_warmup, args.latency_iters, args.latency_repeats)

    params_front_idx = pareto_front([(r["params"], r["proxy_quality_mse"]) for r in results])
    latency_front_idx = pareto_front([(r["latency_ms"], r["proxy_quality_mse"]) for r in results])
    for i in params_front_idx:
        results[i]["pareto_params"] = True
    for i in latency_front_idx:
        results[i]["pareto_latency"] = True
    for i, r in enumerate(results):
        r.setdefault("pareto_params", False)
        r.setdefault("pareto_latency", False)
        # Backward-compatible field: the primary front is now latency-aware.
        r["pareto"] = r["pareto_latency"]

    latency_front = [results[i] for i in latency_front_idx]
    fastest = min(latency_front, key=lambda r: r["latency_ms"])
    best_quality = min(latency_front, key=lambda r: r["proxy_quality_mse"])

    with open(os.path.join(args.out_dir, "search_results.json"), "w") as f:
        json.dump({"selection_basis": "latency-aware Pareto front (RTX GPU latency vs feature MSE)",
                    "teacher_params": teacher_params, "teacher_gmacs": teacher_macs / 1e9,
                    "teacher_latency_ms": teacher_latency_ms, "candidates": results,
                    "selected_fastest": fastest, "selected_best_quality": best_quality,
                    # Retained aliases keep fine-tuning and benchmarking interfaces stable.
                    "selected_smallest": fastest, "selected_largest": best_quality},
                   f, indent=2)

    save_pareto_plot(results, params_front_idx, "params", "Parameters (M)", "Architecture search: parameter Pareto front",
                     os.path.join(args.out_dir, "pareto_front_params.png"))
    save_pareto_plot(results, latency_front_idx, "latency_ms", "Per-step latency (ms, your GPU)",
                     "Architecture search: latency-aware Pareto front", os.path.join(args.out_dir, "pareto_front_latency.png"),
                     fastest, best_quality)
    # This is the main report plot and remains at the historic filename.
    save_pareto_plot(results, latency_front_idx, "latency_ms", "Per-step latency (ms, your GPU)",
                     "Architecture search: latency-aware Pareto front", os.path.join(args.out_dir, "pareto_front.png"),
                     fastest, best_quality)

    print(f"\nSelected FASTEST: depth={fastest['depth']} params={fastest['params']/1e6:.2f}M latency={fastest['latency_ms']:.3f}ms")
    print(f"Selected BEST QUALITY: depth={best_quality['depth']} params={best_quality['params']/1e6:.2f}M latency={best_quality['latency_ms']:.3f}ms")
    print(f"Saved results + plot to {args.out_dir}")


if __name__ == "__main__":
    main()
