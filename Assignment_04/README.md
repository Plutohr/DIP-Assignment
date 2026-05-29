# Assignment 4 - 3D Gaussian Splatting

### In this assignment, I implemented a simplified 3D Gaussian Splatting pipeline with pure PyTorch, used COLMAP to recover camera poses and sparse points, and compared the simplified renderer with the official 3DGS implementation.

### Resources
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)
- [Paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)
- [Official 3DGS Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP Documentation](https://colmap.github.io/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### 1. Structure-from-Motion with COLMAP

The first task uses COLMAP to recover camera intrinsics, camera extrinsics, and sparse 3D points from the multi-view `chair` images. The output sparse model is used as the initialization for 3D Gaussian Splatting.

The generated COLMAP model contains:

```text
data/chair/sparse/0/cameras.bin
data/chair/sparse/0/images.bin
data/chair/sparse/0/points3D.bin
```

For easier debugging, I also exported the sparse model to text format:

```text
data/chair/sparse/0_text/cameras.txt
data/chair/sparse/0_text/images.txt
data/chair/sparse/0_text/points3D.txt
```

### 2. Simplified 3D Gaussian Splatting

The second task implements the core parts of a simplified 3DGS renderer in PyTorch:

1. Constructing 3D covariance matrices from quaternion rotations and scaling parameters.
2. Projecting 3D Gaussian covariances to 2D image-space covariances.
3. Evaluating 2D Gaussian density on image pixels.
4. Rendering RGB images with depth sorting and alpha blending.

Compared with the official implementation, this version intentionally keeps the pipeline simple. It does not include tile-based rasterization, adaptive densification, pruning, or CUDA custom kernels.

### 3. Comparison with Official 3DGS

The third task runs the official 3DGS implementation on the same `chair` dataset and compares the result with the simplified PyTorch version from three aspects:

1. Rendering quality
2. Training speed
3. GPU memory usage

---

## Implementation of Simplified 3DGS and Official 3DGS Comparison

This repository is my implementation of Assignment 4 of DIP.

Simplified PyTorch 3DGS rendering:

<img src="results/task2_h8008/debug_images/epoch_0199.png" alt="simplified 3DGS final debug render" width="800">

Official 3DGS rendering:

<img src="results/task3_official_h8008/renders/00050.png" alt="official 3DGS render" width="800">

## Requirements

The main Python dependencies are:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy opencv-python tqdm
```

Task 1 requires COLMAP:

```bash
colmap -h
```

Task 2 is implemented with PyTorch and CUDA. The final submitted training run was executed on an NVIDIA H800 PCIe GPU through the remote Slurm platform.

Task 3 uses the official 3DGS repository and its CUDA extensions:

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

## Running

To run COLMAP Structure-from-Motion:

```bash
python mvs_with_colmap.py --data_dir data/chair
```

To verify the sparse reconstruction by projecting 3D points back to the images:

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

To train the simplified PyTorch 3DGS model:

```bash
python train.py \
  --colmap_dir data/chair \
  --checkpoint_dir data/chair/checkpoints_h8008 \
  --num_epochs 200 \
  --device cuda \
  --maximum_points 3000 \
  --downsample_factor 8
```

To render a multi-view video from a trained simplified model:

```bash
python render_3dgs_mv.py \
  --colmap_dir data/chair \
  --checkpoint data/chair/checkpoints_h8008/checkpoint_000180.pt \
  --num_frames 240 \
  --fps 30 \
  --device cuda
```

To train the official 3DGS implementation:

```bash
python train.py \
  -s /public/home/ba25001026/3DGS/data/chair \
  -m /public/home/ba25001026/3DGS/task3_official/output/chair_h8008 \
  --disable_viewer \
  --test_iterations -1
```

To render the official 3DGS model:

```bash
python render.py \
  -s /public/home/ba25001026/3DGS/data/chair \
  -m /public/home/ba25001026/3DGS/task3_official/output/chair_h8008 \
  --skip_test
```

The Slurm scripts used for the official 3DGS experiment are saved in:

```text
hpc_task3/
```

## Results

### Task 1: COLMAP Structure-from-Motion

COLMAP successfully registered all 100 images and reconstructed 13,659 sparse 3D points.

| Item | Value |
| --- | ---: |
| Scene | chair |
| Input images | 100 |
| Registered images | 100 |
| Sparse 3D points | 13,659 |
| Output model | `data/chair/sparse/0` |

Projection verification:

<img src="data/chair/projections/r_0.png" alt="COLMAP point projection" width="800">

The recovered sparse reconstruction is saved as:

```text
data/chair/sparse/0
data/chair/sparse/0_text
```

### Task 2: Simplified PyTorch 3DGS

The simplified PyTorch implementation was trained with 3,000 sampled COLMAP points. The input images were downsampled by a factor of 8 to keep the pure PyTorch renderer tractable.

| Item | Value |
| --- | ---: |
| GPU | NVIDIA H800 PCIe |
| Training epochs | 200 |
| Training wall time | 39 min 36 s |
| Input COLMAP points | 13,659 |
| Optimized Gaussians | 3,000 |
| Downsample factor | 8 |
| Last logged loss | 0.0403 |
| Peak GPU memory | 3,237 MB, sampled by 1-epoch probe |

Training progress samples:

<img src="results/task2_h8008/debug_images/epoch_0000.png" alt="simplified 3DGS epoch 0" width="390">
<img src="results/task2_h8008/debug_images/epoch_0100.png" alt="simplified 3DGS epoch 100" width="390">
<img src="results/task2_h8008/debug_images/epoch_0199.png" alt="simplified 3DGS epoch 199" width="390">

The simplified rendering video is saved as:

```text
results/task2_h8008/debug_rendering.mp4
```

The backup checkpoint is saved as:

```text
results/task2_h8008/checkpoints_backup/checkpoint_000180.pt
```

### Task 3: Official 3DGS Comparison

The official 3DGS implementation was trained on the same COLMAP scene with its default 30,000 iterations. The official model uses CUDA rasterization, adaptive densification, and pruning.

| Item | Simplified PyTorch 3DGS | Official 3DGS |
| --- | ---: | ---: |
| GPU | NVIDIA H800 PCIe | NVIDIA H800 PCIe |
| Input scene | `data/chair` | `data/chair` |
| Initial COLMAP points | 13,659 | 13,659 |
| Optimized Gaussians | 3,000 fixed sampled points | 366,474 final Gaussians |
| Training setting | 200 epochs, downsample factor 8 | 30,000 iterations |
| Training wall time | 39 min 36 s | 5 min 46 s |
| Full job wall time | 39 min 36 s | 6 min 15 s |
| Peak GPU memory | 3,237 MB | 2,773 MB |

Official 3DGS rendering samples:

<img src="results/task3_official_h8008/renders/00000.png" alt="official 3DGS render 00000" width="390">
<img src="results/task3_official_h8008/renders/00050.png" alt="official 3DGS render 00050" width="390">
<img src="results/task3_official_h8008/renders/00099.png" alt="official 3DGS render 00099" width="390">

The official rendering video is saved as:

```text
results/task3_official_h8008/official_train_renders.mp4
```

The official final point cloud is saved as:

```text
results/task3_official_h8008/point_cloud/point_cloud.ply
```

The detailed comparison summary is saved as:

```text
results/task3_official_h8008/comparison_summary.md
```

## Discussion

The simplified PyTorch implementation correctly follows the core mathematical pipeline of 3D Gaussian Splatting: 3D covariance construction, projection to 2D covariance, Gaussian evaluation, and alpha blending. It is useful for understanding the algorithm, but the reconstruction quality is limited because it uses a fixed set of 3,000 Gaussians and does not perform densification.

The official 3DGS implementation produces a much denser and cleaner reconstruction. Starting from the same COLMAP sparse point cloud, it increases the representation to 366,474 Gaussians through adaptive densification and pruning. This gives better surface coverage and sharper rendering details.

The official implementation is also much faster. In this experiment, it finished the default 30,000-iteration training in 346 seconds, while the simplified PyTorch implementation took 39 minutes and 36 seconds for 200 epochs on downsampled images. The main reason is that the official implementation uses optimized CUDA kernels and tile-based rasterization, while the simplified version performs dense tensor operations directly in PyTorch.

The GPU memory comparison shows the same trend. Even though the official implementation trains a much larger Gaussian set, its sampled peak GPU memory was 2,773 MB, lower than the simplified implementation's 3,237 MB probe. This is because the official renderer avoids many large intermediate tensors and uses visibility-aware CUDA rasterization.

## Acknowledgement

Thanks for the ideas and tools from 3D Gaussian Splatting, COLMAP, PyTorch, and the official GraphDECO 3DGS implementation.
