#!/usr/bin/env bash
#SBATCH --job-name=official_3dgs_h800
#SBATCH --partition=GPU_H800_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/ba25001026/3DGS/task3_official/logs/%x_%j.out
#SBATCH --error=/public/home/ba25001026/3DGS/task3_official/logs/%x_%j.err
#SBATCH --chdir=/public/home/ba25001026/3DGS/task3_official/gaussian-splatting

bash /public/home/ba25001026/3DGS/task3_official/run_official_h800.sh
