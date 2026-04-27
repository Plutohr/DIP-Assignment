"""CUDA-only Bundle Adjustment for Assignment 3.

The script optimizes a shared focal length, per-view extrinsics, and 3D points
from the provided 2D observations. It intentionally refuses to run without CUDA.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-dip-ba")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SIZE = 1024.0
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-only PyTorch Bundle Adjustment")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--batch-points", type=int, default=4096)
    parser.add_argument("--lr-points", type=float, default=2e-2)
    parser.add_argument("--lr-camera", type=float, default=1e-3)
    parser.add_argument("--lr-focal", type=float, default=1e-1)
    parser.add_argument("--weight-decay-points", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This assignment run is CUDA-only; CPU fallback is disabled.")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError(f"Device must be CUDA, got {device_name!r}.")
    torch.empty(1, device=device)
    return device


def load_observations(
    data_dir: Path,
    device: torch.device,
    max_views: int | None,
    max_points: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    points2d_npz = np.load(data_dir / "points2d.npz")
    keys = sorted(points2d_npz.files)
    if max_views is not None:
        keys = keys[:max_views]

    obs_np = np.stack([points2d_npz[k][:, :2] for k in keys], axis=0).astype(np.float32)
    mask_np = np.stack([points2d_npz[k][:, 2] > 0.5 for k in keys], axis=0)
    colors_np = np.load(data_dir / "points3d_colors.npy").astype(np.float32)

    if max_points is not None:
        obs_np = obs_np[:, :max_points]
        mask_np = mask_np[:, :max_points]
        colors_np = colors_np[:max_points]

    obs = torch.from_numpy(obs_np).to(device=device)
    mask = torch.from_numpy(mask_np).to(device=device)
    colors = torch.from_numpy(colors_np).to(device=device)
    return obs, mask, colors, keys


def inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def initialize_points(obs: torch.Tensor, mask: torch.Tensor, focal_init: float, depth: float) -> torch.Tensor:
    visible = mask.float()
    counts = visible.sum(dim=0).clamp_min(1.0)
    mean_xy = (obs * visible[..., None]).sum(dim=0) / counts[:, None]
    x = (mean_xy[:, 0] - CX) * depth / focal_init
    y = (CY - mean_xy[:, 1]) * depth / focal_init
    z = 0.03 * torch.randn_like(x)
    points = torch.stack([x, y, z], dim=1)
    missing = mask.sum(dim=0) == 0
    if missing.any():
        points[missing] = 0.03 * torch.randn((int(missing.sum()), 3), device=obs.device)
    return points


def initialize_cameras(num_views: int, depth: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    euler = torch.zeros((num_views, 3), dtype=torch.float32, device=device)
    if num_views > 1:
        euler[:, 1] = torch.linspace(math.radians(-70.0), math.radians(70.0), num_views, device=device)
    trans = torch.zeros((num_views, 3), dtype=torch.float32, device=device)
    trans[:, 2] = -depth
    return euler, trans


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rz @ ry @ rx


def project(points: torch.Tensor, euler: torch.Tensor, trans: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    rotations = euler_xyz_to_matrix(euler)
    camera_points = torch.einsum("vij,nj->vni", rotations, points) + trans[:, None, :]
    z = camera_points[..., 2].clamp_max(-1e-4)
    u = -focal * camera_points[..., 0] / z + CX
    v = focal * camera_points[..., 1] / z + CY
    return torch.stack([u, v], dim=-1)


def reprojection_loss(
    points: torch.Tensor,
    euler: torch.Tensor,
    trans: torch.Tensor,
    focal: torch.Tensor,
    obs_batch: torch.Tensor,
    mask_batch: torch.Tensor,
) -> torch.Tensor:
    pred = project(points, euler, trans, focal)
    squared = ((pred - obs_batch) ** 2).sum(dim=-1)
    valid = mask_batch.float()
    return (squared * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_full_loss(
    points: torch.Tensor,
    euler: torch.Tensor,
    trans: torch.Tensor,
    focal: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    batch_points: int,
) -> float:
    total_squared = torch.zeros((), dtype=torch.float32, device=points.device)
    total_valid = torch.zeros((), dtype=torch.float32, device=points.device)
    for start in range(0, points.shape[0], batch_points):
        end = min(start + batch_points, points.shape[0])
        pred = project(points[start:end], euler, trans, focal)
        squared = ((pred - obs[:, start:end, :]) ** 2).sum(dim=-1)
        valid = mask[:, start:end].float()
        total_squared += (squared * valid).sum()
        total_valid += valid.sum()
    return float((total_squared / total_valid.clamp_min(1.0)).detach().cpu())


def write_obj(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            r, g, b = np.clip(c, 0.0, 1.0)
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {r:.7f} {g:.7f} {b:.7f}\n")


def save_loss_curve(path: Path, rows: list[dict[str, float]]) -> None:
    iters = [row["iter"] for row in rows]
    losses = [row["loss"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(iters, losses, linewidth=1.6)
    plt.xlabel("Iteration")
    plt.ylabel("Mean squared reprojection error")
    plt.title("Bundle Adjustment Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_log_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "loss", "focal"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = require_cuda(args.device)
    device_name = torch.cuda.get_device_name(device)
    print(f"Using device: {device_name}")

    obs, mask, colors, keys = load_observations(args.data_dir, device, args.max_views, args.max_points)
    num_views, num_points = obs.shape[:2]
    print(f"Loaded observations: {num_views} views, {num_points} points")

    depth = 2.5
    focal_init = IMAGE_SIZE / (2.0 * math.tan(math.radians(60.0) / 2.0))

    points_param = torch.nn.Parameter(initialize_points(obs, mask, focal_init, depth))
    euler_init, trans_init = initialize_cameras(num_views, depth, device)
    euler_param = torch.nn.Parameter(euler_init)
    trans_param = torch.nn.Parameter(trans_init)
    raw_focal = torch.nn.Parameter(torch.tensor(inverse_softplus(focal_init), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [points_param], "lr": args.lr_points, "weight_decay": args.weight_decay_points},
            {"params": [euler_param, trans_param], "lr": args.lr_camera},
            {"params": [raw_focal], "lr": args.lr_focal},
        ]
    )

    rows: list[dict[str, float]] = []
    all_indices = torch.arange(num_points, device=device)
    batch_points = min(args.batch_points, num_points)

    for step in range(1, args.iters + 1):
        if batch_points < num_points:
            point_idx = all_indices[torch.randperm(num_points, device=device)[:batch_points]]
        else:
            point_idx = all_indices

        focal = F.softplus(raw_focal) + 1e-6
        loss = reprojection_loss(
            points_param[point_idx],
            euler_param,
            trans_param,
            focal,
            obs[:, point_idx, :],
            mask[:, point_idx],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.iters:
            with torch.no_grad():
                focal_value = float((F.softplus(raw_focal) + 1e-6).detach().cpu())
                loss_value = float(loss.detach().cpu())
            rows.append({"iter": step, "loss": loss_value, "focal": focal_value})
            print(f"iter {step:05d} | loss {loss_value:.6f} | focal {focal_value:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_np = points_param.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    focal_np = float((F.softplus(raw_focal) + 1e-6).detach().cpu())
    final_full_loss = evaluate_full_loss(
        points_param,
        euler_param,
        trans_param,
        F.softplus(raw_focal) + 1e-6,
        obs,
        mask,
        batch_points,
    )
    euler_np = euler_param.detach().cpu().numpy()
    trans_np = trans_param.detach().cpu().numpy()

    write_obj(args.output_dir / "optimized_points3d.obj", points_np, colors_np)
    save_loss_curve(args.output_dir / "loss_curve.png", rows)
    save_log_csv(args.output_dir / "train_log.csv", rows)
    np.savez(
        args.output_dir / "optimized_params.npz",
        points3d=points_np,
        colors=colors_np,
        focal=np.array(focal_np, dtype=np.float32),
        euler=euler_np,
        trans=trans_np,
        view_keys=np.array(keys),
    )
    final_loss = rows[-1]["loss"] if rows else float("nan")
    (args.output_dir / "README_task1_results.md").write_text(
        "\n".join(
            [
                "# Task 1 Bundle Adjustment Results",
                "",
                f"- Device: {device_name}",
                f"- Views: {num_views}",
                f"- Points: {num_points}",
                f"- Iterations: {args.iters}",
                f"- Batch points: {batch_points}",
                f"- Final logged loss: {final_loss:.6f}",
                f"- Final full loss: {final_full_loss:.6f}",
                f"- Final focal length: {focal_np:.6f}",
                "",
                "Generated files:",
                "- `optimized_points3d.obj`",
                "- `loss_curve.png`",
                "- `train_log.csv`",
                "- `optimized_params.npz`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Final full loss: {final_full_loss:.6f}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
