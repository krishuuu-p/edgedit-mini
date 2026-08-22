# EdgeDiT-Mini

EdgeDiT-Mini is a reduced-scale implementation of the core workflow in
*EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image
Generation*. It is designed to run on a laptop GPU while preserving the paper's
main stages: hardware-aware block surgeries, feature-wise knowledge distillation
(FwKD), candidate assembly, latency-aware Pareto selection, end-to-end
fine-tuning, and evaluation.

## Scope and correspondence to the paper

| Aspect | EdgeDiT paper | This implementation |
|---|---|---|
| Dataset | ImageNet, 256x256 VAE latents | FashionMNIST, 28x28 pixels |
| Teacher | DiT-XL/2, 28 layers, width 1152, about 675M params | Tiny-DiT, 6 layers, width 128, 1.875M params |
| Surgery choices | Block removal, MLP-ratio change, hidden-dimension reduction | Same three surrogate operations |
| Candidate selection | EHVI-based MOBO over FID and device latency | Sampled latency-aware Pareto front over feature MSE and RTX GPU latency |
| Quality metric | ImageNet FID-50K and related metrics | Proxy-FID using FashionMNIST-CNN features |
| Hardware metric | Apple/Qualcomm mobile NPUs | Measured RTX 3050 denoising-step latency |

The experiments reproduce the paper's methodological objective at a reduced
scale. Proxy-FID and RTX 3050 timing values must not be numerically compared
with the paper's ImageNet FID-50K or mobile-NPU measurements.

## Setup

```bash
pip install -r requirements.txt
```

FashionMNIST downloads automatically on the first run. CUDA is used by default
when PyTorch detects it; otherwise the scripts run on CPU.

## One-command experiments

Run the complete pipeline with:

```bash
python run_all.py --preset quality
```

The pipeline performs seven stages:

1. Train the Tiny-DiT teacher.
2. Distil block-removal, MLP-ratio, and hidden-dimension surrogates.
3. Measure candidate latency, construct Pareto fronts, and select two candidates.
4. Fine-tune EdgeDiT-small and EdgeDiT-large.
5. Benchmark parameters, estimated compute, latency, and Proxy-FID.
6. Generate surgery and FwKD ablation figures.
7. Produce compact three-sample report figures.

### Presets

| Preset | Teacher steps | Surrogate steps | Candidates | Fine-tune steps | Evaluation | Typical RTX 3050 time |
|---|---:|---:|---:|---:|---|---:|
| `quick` | 2K | 200 | 15 | 1K | 100 samples, 100 DDIM steps | about 7 min |
| `standard` | 12K | 800 | 30 | 6K | 200 samples, 250 DDIM steps | about 20 min |
| `quality` | 30K | 1.6K | 60 | 15K | 500 samples, 500 DDIM steps | about 50 min |

`quick` is a smoke test. Use `quality` for the strongest report-ready results.
Times depend on the GPU and should be treated as estimates.

### Run management and resuming

Each run is isolated as:

```text
runs/<preset>/run<number>/
```

Without `--run_number`, the runner creates the next independent run:

```bash
python run_all.py --preset standard
```

To resume a specific interrupted run, supply its number:

```bash
python run_all.py --preset standard --run_number 1
```

Completed stages are skipped automatically. Use `--force` only when deliberately
regenerating all stages in that run. To omit the random-initialization FwKD
ablation, add `--skip_no_kd_ablation`.

## Outputs

For a completed run at `runs/quality/run1/`:

| Location | Contents | Report use |
|---|---|---|
| `teacher/` | Teacher checkpoint, samples, loss log | Baseline samples and learning evidence |
| `search/search_results.json` | Candidate layouts, parameters, latency, feature MSE, selected models | Architecture details |
| `search/pareto_front.png` | Primary latency-aware Pareto plot | Hardware-aware search figure |
| `search/pareto_front_latency.png` | Explicit latency-vs-feature-MSE Pareto plot | Same as above |
| `search/pareto_front_params.png` | Parameter-vs-feature-MSE comparison | Supplemental analysis |
| `finetuned/` | Selected model checkpoints, samples, loss logs | Teacher/small/large comparisons |
| `benchmark/benchmark_table.csv` | Final parameter, GMAC, GFLOP, latency, and Proxy-FID table | Main quantitative table |
| `figures/` | Original ten-sample surgery and FwKD grids | Full supporting evidence |
| `report_figures/` | Compact three-sample grids and Pareto plot | Report and presentation figures |

The report grids select FashionMNIST classes 0, 5, and 9: T-shirt/top, sandal,
and ankle boot. Creating them only crops existing PNGs; it does not retrain or
resample models.

## What is selected during search?

Each candidate combines original teacher blocks with distilled surrogates. Search
measures a single-image denoising-step latency on the selected device and compares
it with candidate feature MSE relative to the teacher. The primary Pareto front is
therefore latency-aware.

- **EdgeDiT-small** is the fastest non-dominated candidate.
- **EdgeDiT-large** is the lowest-feature-error candidate on the latency-aware
  Pareto front.

The final quality metric is evaluated only for the teacher and the two selected,
fine-tuned models. The Proxy-FID classifier runs on CPU by default to avoid memory
pressure on laptop GPUs; diffusion generation and latency measurement remain on
the requested device. Pass `--metric_device cuda` directly to `benchmark.py`
only when sufficient GPU memory is available.

## Citation

```text
S. Kodavanti, M. Arveti, S. Vajrala, S. Miriyala, and V. N. R.
“EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device
Image Generation.” arXiv:2603.28405, 2026.
```
