#!/usr/bin/env bash
#SBATCH --job-name=task2_mem_probe
#SBATCH --partition=GPU_H800_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/ba25001026/3DGS/task3_official/logs/%x_%j.out
#SBATCH --error=/public/home/ba25001026/3DGS/task3_official/logs/%x_%j.err
#SBATCH --chdir=/public/home/ba25001026/3DGS

bash /public/home/ba25001026/3DGS/task3_official/task2_mem_probe_h800.sh
