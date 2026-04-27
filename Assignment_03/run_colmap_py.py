"""Run Assignment 3 Task 2 reconstruction with pycolmap.

This avoids requiring a system-wide COLMAP binary. Outputs are written under
outputs/task2_colmap by default.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP reconstruction via pycolmap")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_colmap"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-image-size", type=int, default=1024)
    return parser.parse_args()


def get_device(name: str) -> pycolmap.Device:
    if name == "cuda":
        return pycolmap.Device.cuda
    if name == "cpu":
        return pycolmap.Device.cpu
    return pycolmap.Device.auto


@contextlib.contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def write(self, data: str) -> int:
                original_stdout.write(data)
                log.write(data)
                log.flush()
                return len(data)

            def flush(self) -> None:
                original_stdout.flush()
                log.flush()

        tee = Tee()
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "sparse").mkdir(parents=True)
    (output_dir / "dense").mkdir(parents=True)


def run(args: argparse.Namespace) -> None:
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    ensure_clean_output(args.output_dir, args.overwrite)
    database_path = args.output_dir / "database.db"
    sparse_dir = args.output_dir / "sparse"
    dense_dir = args.output_dir / "dense"
    device = get_device(args.device)

    print(f"pycolmap version: {pycolmap.__version__}")
    print(f"Images: {args.image_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device request: {args.device}")

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.default_focal_length_factor = 1.2

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = args.max_image_size
    extraction_options.use_gpu = args.device != "cpu"
    extraction_options.gpu_index = "0"

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = args.device != "cpu"
    matching_options.gpu_index = "0"
    matching_options.guided_matching = True

    print("\n=== Step 1: Feature Extraction ===")
    pycolmap.extract_features(
        database_path,
        args.image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )

    print("\n=== Step 2: Exhaustive Matching ===")
    pycolmap.match_exhaustive(
        database_path,
        matching_options=matching_options,
        device=device,
    )

    print("\n=== Step 3: Sparse Reconstruction / Mapper ===")
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.ba_use_gpu = args.device != "cpu"
    mapper_options.ba_gpu_index = "0"
    mapper_options.multiple_models = False
    mapper_options.extract_colors = True
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        args.image_dir,
        sparse_dir,
        options=mapper_options,
    )
    if not reconstructions:
        raise RuntimeError("Sparse reconstruction failed: no models were produced.")

    best_model_id = max(
        reconstructions,
        key=lambda idx: reconstructions[idx].num_reg_images(),
    )
    sparse_model_dir = sparse_dir / str(best_model_id)
    reconstruction = reconstructions[best_model_id]
    print(f"Best sparse model: {best_model_id}")
    print(f"Registered images: {reconstruction.num_reg_images()}")
    print(f"3D points: {reconstruction.num_points3D()}")

    if args.skip_dense:
        print("\nSkipping dense reconstruction.")
        fused_path = None
        dense_status = "skipped"
    else:
        print("\n=== Step 4: Image Undistortion ===")
        pycolmap.undistort_images(
            dense_dir,
            sparse_model_dir,
            args.image_dir,
            output_type="COLMAP",
        )

        print("\n=== Step 5: Dense Reconstruction / Patch Match Stereo ===")
        patch_options = pycolmap.PatchMatchOptions()
        patch_options.max_image_size = args.max_image_size
        patch_options.gpu_index = "0" if args.device != "cpu" else "-1"
        try:
            pycolmap.patch_match_stereo(
                dense_dir,
                options=patch_options,
            )

            print("\n=== Step 6: Stereo Fusion ===")
            fused_path = dense_dir / "fused.ply"
            fusion_options = pycolmap.StereoFusionOptions()
            fusion_options.max_image_size = args.max_image_size
            pycolmap.stereo_fusion(
                fused_path,
                dense_dir,
                options=fusion_options,
                output_type="PLY",
            )
            dense_status = "completed"
            print(f"Dense fused point cloud: {fused_path}")
        except Exception as exc:
            fused_path = None
            dense_status = f"failed: {exc}"
            print(f"Dense reconstruction failed, keeping sparse result only: {exc}")

    sparse_ply = args.output_dir / "sparse_points.ply"
    reconstruction.export_PLY(sparse_ply)
    print(f"Sparse point cloud: {sparse_ply}")

    summary_lines = [
        "# Task 2 COLMAP Results",
        "",
        f"- pycolmap version: {pycolmap.__version__}",
        f"- Image directory: `{args.image_dir}`",
        f"- Output directory: `{args.output_dir}`",
        f"- Device request: `{args.device}`",
        f"- Registered images: {reconstruction.num_reg_images()}",
        f"- Sparse 3D points: {reconstruction.num_points3D()}",
        f"- Sparse model: `{sparse_model_dir}`",
        f"- Sparse PLY: `{sparse_ply}`",
        f"- Dense status: {dense_status}",
    ]
    preview_path = args.output_dir / "sparse_preview.png"
    if preview_path.exists():
        summary_lines.append(f"- Sparse preview: `{preview_path}`")
    if fused_path is not None:
        summary_lines.append(f"- Dense fused PLY: `{fused_path}`")
    else:
        summary_lines.append("- Dense reconstruction: skipped")
    (args.output_dir / "README_task2_results.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n=== Done ===")


def main() -> None:
    args = parse_args()
    log_path = args.output_dir / "run_log.txt"
    with tee_stdout(log_path):
        run(args)


if __name__ == "__main__":
    main()
