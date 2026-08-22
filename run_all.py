import argparse
import json
import os
import subprocess
import sys


PRESETS = {
    "quick": {
        "teacher_steps": 2_000,
        "surrogate_steps": 200,
        "candidates": 15,
        "finetune_steps": 1_000,
        "fid_samples": 100,
        "sample_steps": 100,
        "no_kd_steps": 500,
    },
    "standard": {
        "teacher_steps": 12_000,
        "surrogate_steps": 800,
        "candidates": 30,
        "finetune_steps": 6_000,
        "fid_samples": 200,
        "sample_steps": 250,
        "no_kd_steps": 2_000,
    },
    "quality": {
        "teacher_steps": 30_000,
        "surrogate_steps": 1_600,
        "candidates": 60,
        "finetune_steps": 15_000,
        "fid_samples": 500,
        "sample_steps": 500,
        "no_kd_steps": 4_000,
    },
}


def run_stage(name, command, expected_outputs, force):
    complete = all(os.path.isfile(path) for path in expected_outputs)
    if complete and not force:
        print(f"\n[{name}] already complete; skipping (use --force to rerun).")
        return
    print(f"\nStarting stage {name}...")
    print(" ".join(command))
    subprocess.run(command, check=True)


def next_run_number(preset):
    preset_dir = os.path.join("./runs", preset)
    if not os.path.isdir(preset_dir):
        return 1
    numbers = []
    for entry in os.listdir(preset_dir):
        if entry.startswith("run") and entry[3:].isdigit():
            numbers.append(int(entry[3:]))
    return max(numbers, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="One-command EdgeDiT-Mini reproduction runner")
    parser.add_argument("--preset", choices=PRESETS, default="standard",
                        help="quick is a smoke test; standard is report-ready; quality prioritizes generation quality")
    parser.add_argument("--run_number", type=int,
                        help="resume this experiment number; omit to start a new run")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="rerun stages even when their outputs exist")
    parser.add_argument("--skip_no_kd_ablation", action="store_true",
                        help="omit the random-initialization FwKD ablation to save time")
    args = parser.parse_args()
    if args.run_number is not None and args.run_number < 1:
        parser.error("--run_number must be at least 1")
    if args.run_number is None:
        args.run_number = next_run_number(args.preset)
        print(f"No run number supplied; starting independent run {args.run_number}.")
    cfg = PRESETS[args.preset]
    run_dir = os.path.join("./runs", args.preset, f"run{args.run_number}")
    teacher_dir = os.path.join(run_dir, "teacher")
    teacher_ckpt = os.path.join(teacher_dir, "teacher_ckpt.pt")
    surrogates_path = os.path.join(run_dir, "surrogates.pt")
    search_dir = os.path.join(run_dir, "search")
    search_results = os.path.join(search_dir, "search_results.json")
    finetuned_dir = os.path.join(run_dir, "finetuned")
    benchmark_dir = os.path.join(run_dir, "benchmark")
    figures_dir = os.path.join(run_dir, "figures")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    run_config = {"preset": args.preset, "run_number": args.run_number,
                  "skip_no_kd_ablation": args.skip_no_kd_ablation}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            previous_config = json.load(f)
        if previous_config != run_config:
            print("Previous outputs use a different preset; rerunning all stages for a consistent experiment.")
            args.force = True
    elif os.path.isfile(teacher_ckpt):
        print("Existing outputs have no experiment manifest; rerunning all stages for a consistent experiment.")
        args.force = True
    os.makedirs(run_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(run_config, f, indent=2)
    py = sys.executable
    device_arg = ["--device", args.device]

    print(f"Running preset={args.preset}, run={args.run_number} on device={args.device}")
    print(f"Outputs: {run_dir}")
    print("This is resumable: interrupt safely and run the same command again.")

    run_stage(
        "1/7 Train teacher",
        [py, "train_teacher.py", "--steps", str(cfg["teacher_steps"]), "--out_dir", teacher_dir, *device_arg],
        [teacher_ckpt, os.path.join(teacher_dir, "loss_log.json")], args.force,
    )
    run_stage(
        "2/7 Distil surrogate blocks",
        [py, "distill_surrogates.py", "--teacher_ckpt", teacher_ckpt,
         "--steps_per_surrogate", str(cfg["surrogate_steps"]), "--out_path", surrogates_path, *device_arg],
        [surrogates_path], args.force,
    )
    run_stage(
        "3/7 Search and select architectures",
        [py, "search.py", "--teacher_ckpt", teacher_ckpt, "--surrogates", surrogates_path,
         "--num_candidates", str(cfg["candidates"]), "--out_dir", search_dir, *device_arg],
        [search_results, os.path.join(search_dir, "pareto_front.png")], args.force,
    )
    run_stage(
        "4/7 Fine-tune selected architectures",
        [py, "finetune_candidates.py", "--teacher_ckpt", teacher_ckpt, "--surrogates", surrogates_path,
         "--search_results", search_results, "--steps", str(cfg["finetune_steps"]),
         "--out_dir", finetuned_dir, *device_arg],
        [os.path.join(finetuned_dir, "EdgeDiT-small", "ckpt.pt"),
         os.path.join(finetuned_dir, "EdgeDiT-large", "ckpt.pt")], args.force,
    )
    run_stage(
        "5/7 Benchmark models",
        [py, "benchmark.py", "--teacher_ckpt", teacher_ckpt, "--surrogates", surrogates_path,
         "--search_results", search_results, "--finetuned_dir", finetuned_dir,
         "--out_dir", benchmark_dir, "--n_fid_samples", str(cfg["fid_samples"]),
         "--sample_steps", str(cfg["sample_steps"]), *device_arg],
        [os.path.join(benchmark_dir, "benchmark_table.csv"),
         os.path.join(benchmark_dir, "benchmark_table.json")], args.force,
    )

    figure_command = [py, "make_ablation_figures.py", "--teacher_ckpt", teacher_ckpt,
                      "--surrogates", surrogates_path, "--search_results", search_results,
                      "--finetuned_dir", finetuned_dir, "--out_dir", figures_dir,
                      "--sample_steps", str(cfg["sample_steps"]), "--no_kd_steps", str(cfg["no_kd_steps"]), *device_arg]
    expected_figures = [os.path.join(figures_dir, "fig_block_removal_merge1pairs.png"),
                        os.path.join(figures_dir, "fig_mlp_ratio_r2_reduced.png"),
                        os.path.join(figures_dir, "fig_hidden_dim_d64_reduced.png")]
    if args.skip_no_kd_ablation:
        figure_command.append("--skip_no_kd_ablation")
    else:
        expected_figures += [os.path.join(figures_dir, "fig_ablation_WITH_fwkd.png"),
                             os.path.join(figures_dir, "fig_ablation_WITHOUT_fwkd.png")]
    run_stage("6/7 Generate ablation figures", figure_command, expected_figures, args.force)

    report_figures_dir = os.path.join(run_dir, "report_figures")
    run_stage(
        "7/7 Create compact report figures",
        [py, "make_report_grids.py", "--run_dir", run_dir],
        [os.path.join(report_figures_dir, "teacher_3samples.png"),
         os.path.join(report_figures_dir, "edgedit_small_3samples.png"),
         os.path.join(report_figures_dir, "edgedit_large_3samples.png"),
         os.path.join(report_figures_dir, "pareto_front.png")],
        args.force,
    )

    print(f"\nComplete. Report outputs are in {run_dir}.")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
