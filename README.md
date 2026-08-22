# EdgeDiT-Mini

This project is a reduced-scale implementation of the EdgeDiT pipeline for
efficient diffusion transformers. It uses FashionMNIST and a small DiT model so
the full experiment can run on a laptop GPU.

The pipeline includes block removal, MLP-ratio reduction, hidden-dimension
reduction, feature-wise knowledge distillation, architecture search, fine-tuning,
benchmarking, and ablation figures.

## Setup

```bash
pip install -r requirements.txt
```

FashionMNIST is downloaded automatically when required. CUDA is used if it is
available.

## Run an experiment

```bash
python run_all.py --preset quality
```

Available presets:

| Preset | Use | Approximate time on RTX 3050 |
|---|---|---:|
| `quick` | Smoke test | 7 min |
| `standard` | Normal experiment | 20 min |
| `quality` | Best report-ready output | 50 min |

Each run follows this sequence:

1. Train the Tiny-DiT teacher.
2. Train surrogate blocks with feature-wise knowledge distillation.
3. Search candidate architectures using latency and feature MSE.
4. Fine-tune EdgeDiT-small and EdgeDiT-large.
5. Benchmark the selected models and generate figures.

Runs are saved under `runs/<preset>/run<number>/`. If `--run_number` is omitted,
a new run number is created automatically:

```bash
python run_all.py --preset standard
```

Resume an interrupted run by specifying its number:

```bash
python run_all.py --preset standard --run_number 1
```

Use `--skip_no_kd_ablation` to skip the random-initialization FwKD comparison.

## Outputs

Each completed run contains:

| Folder | Contents |
|---|---|
| `teacher/` | Teacher checkpoint, samples, and loss log |
| `search/` | Candidate results and latency-aware Pareto plots |
| `finetuned/` | EdgeDiT-small and EdgeDiT-large checkpoints and samples |
| `benchmark/` | Parameters, compute, latency, and Proxy-FID table |
| `figures/` | Full surgery and FwKD ablation grids |
| `report_figures/` | Compact three-sample images for reports and slides |

`pareto_front.png` is the main latency-aware Pareto plot. The compact images use
FashionMNIST classes T-shirt/top, sandal, and ankle boot.

## Scope

The paper uses ImageNet, DiT-XL/2, FID-50K, and mobile NPUs. This project uses
FashionMNIST, Tiny-DiT, Proxy-FID, and RTX 3050 latency. It reproduces the core
workflow at smaller scale, so the reported numbers should not be compared directly
with the paper's ImageNet or mobile-device numbers.

## Reference

S. Kodavanti et al., *EdgeDiT: Hardware-Aware Diffusion Transformers for
Efficient On-Device Image Generation*, arXiv:2603.28405, 2026.
