#!/bin/bash
#SBATCH --job-name=emotion
#SBATCH --ntasks=1
#SBATCH --nodelist=n19
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output slurm_log/slurm.%J.out
#SBATCH --error slurm_log/slurm.%J.err
#SBATCH --time=48:00:00

export WANDB_API_KEY=495a5b4509ce4b1ae15055b8f810dc296d4fa6fa
CUDA_VISIBLE_DEVICES=0 python main.py --new_sweep