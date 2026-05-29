# Task 3: Simplified 3DGS vs Official 3DGS

## Experiment Setup

Both methods use the same `chair` scene reconstructed by COLMAP.

| Item | Simplified PyTorch 3DGS | Official 3DGS |
|---|---:|---:|
| GPU | NVIDIA H800 PCIe | NVIDIA H800 PCIe |
| Slurm partition | GPU_H800_8 | GPU_H800_8 |
| Input scene | `data/chair` | `data/chair` |
| Initial COLMAP points | 13,659 | 13,659 |
| Optimized points used | 3,000 sampled points | Full point cloud with densification |
| Training setting | 200 epochs, downsample factor 8 | 30,000 default iterations |

## Quantitative Comparison

| Metric | Simplified PyTorch 3DGS | Official 3DGS |
|---|---:|---:|
| Training wall time | 39 min 36 s | 5 min 46 s training, 6 min 15 s full job |
| Peak GPU memory | 3,237 MB sampled by 1-epoch probe | 2,773 MB |
| Rendering output | `../task2_h8008/debug_rendering.mp4` | `official_train_renders.mp4` |
| Final point cloud | Not exported as official `.ply` | `point_cloud/point_cloud.ply` |

## Observations

The official 3DGS implementation is significantly faster and produces a denser, cleaner reconstruction. The simplified PyTorch implementation is useful for understanding the core math, but it directly evaluates Gaussian contributions in PyTorch and does not include the optimized CUDA rasterizer used by the official implementation.

The official implementation also performs adaptive densification and pruning. Starting from the same COLMAP sparse points, it dynamically adds and removes Gaussians according to image-space gradients and opacity, which improves surface coverage and fine details. The simplified implementation keeps a fixed sampled set of 3,000 Gaussians, so it cannot recover dense geometry with the same quality.

The measured GPU memory is comparable, but the official implementation uses less memory in this experiment despite training a stronger model. This mainly comes from its tile-based CUDA rasterization, efficient visibility handling, and optimized custom kernels. The simplified PyTorch renderer keeps larger intermediate tensors in the high-level framework, so memory usage is less efficient.

## Local Result Files

- Official render video: `official_train_renders.mp4`
- Official representative renders: `renders/00000.png`, `renders/00025.png`, `renders/00050.png`, `renders/00075.png`, `renders/00099.png`
- Official final point cloud: `point_cloud/point_cloud.ply`
- Official training summary: `logs/summary_13705.txt`
- Official GPU memory log: `logs/nvidia_smi_13705.csv`
- Simplified GPU memory probe: `logs/task2_mem_probe_13715.txt`
