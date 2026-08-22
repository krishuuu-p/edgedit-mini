"""Create compact, report-ready three-sample grids from a completed run.

The source 10-sample grids are preserved.  This script selects class columns
0, 5, and 9 (T-shirt/top, Sandal, Ankle boot) and writes compact copies to
<run_dir>/report_figures.  It performs no model inference or training.
"""
import argparse
import glob
import os
import shutil

from PIL import Image


CLASS_COLUMNS = (0, 5, 9)
SOURCE_COLUMNS = 10
PADDING = 2  # torchvision.utils.make_grid default padding


def latest_sample(directory):
    paths = glob.glob(os.path.join(directory, "sample_step*.png"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def compact_grid(source_path, output_path):
    """Select three tiles from a one-row torchvision grid."""
    image = Image.open(source_path).convert("RGB")
    tile_width = (image.width - PADDING * (SOURCE_COLUMNS + 1)) // SOURCE_COLUMNS
    tile_height = image.height - 2 * PADDING
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError(f"Unexpected grid dimensions for {source_path}: {image.size}")

    result = Image.new("RGB", (3 * tile_width + 4 * PADDING, tile_height + 2 * PADDING), "white")
    for output_index, source_index in enumerate(CLASS_COLUMNS):
        left = PADDING + source_index * (tile_width + PADDING)
        tile = image.crop((left, PADDING, left + tile_width, PADDING + tile_height))
        destination_x = PADDING + output_index * (tile_width + PADDING)
        result.paste(tile, (destination_x, PADDING))
    result.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create compact three-sample report grids")
    parser.add_argument("--run_dir", required=True, help="completed runs/<preset>/run<number> directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = os.path.join(run_dir, "report_figures")
    os.makedirs(output_dir, exist_ok=True)

    sources = {
        "teacher_3samples.png": latest_sample(os.path.join(run_dir, "teacher")),
        "edgedit_small_3samples.png": latest_sample(os.path.join(run_dir, "finetuned", "EdgeDiT-small")),
        "edgedit_large_3samples.png": latest_sample(os.path.join(run_dir, "finetuned", "EdgeDiT-large")),
    }
    figures_dir = os.path.join(run_dir, "figures")
    for source_path in glob.glob(os.path.join(figures_dir, "*.png")):
        sources[os.path.basename(source_path).replace(".png", "_3samples.png")] = source_path

    created = []
    for output_name, source_path in sources.items():
        if source_path is None:
            continue
        output_path = os.path.join(output_dir, output_name)
        compact_grid(source_path, output_path)
        created.append(output_path)

    # The Pareto plot is already report-sized and should remain uncropped.
    pareto = os.path.join(run_dir, "search", "pareto_front.png")
    if os.path.isfile(pareto):
        pareto_output = os.path.join(output_dir, "pareto_front.png")
        shutil.copy2(pareto, pareto_output)
        created.append(pareto_output)

    print(f"Created {len(created)} report figures in {output_dir}")
    print("Selected classes: 0=T-shirt/top, 5=Sandal, 9=Ankle boot")


if __name__ == "__main__":
    main()
