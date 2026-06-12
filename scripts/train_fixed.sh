#!/bin/bash
#SBATCH --job-name=draw_fixed
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/fixed_%j.log
#SBATCH --error=logs/fixed_%j.err
#SBATCH --cpus-per-task=8

# Load modules
module load python
conda activate drawspeech

# Disable wandb
export WANDB_MODE=offline
export WANDB_ANONYMOUS=must

# Set paths
export PYTHONPATH=$PYTHONPATH:/home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch:/home/hpc/iwi5/iwi5408h/taming-transformers

# Go to project directory
cd /home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch

# Train with fixed checkpoint
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py \
-c drawspeech/config/drawspeech_ljspeech_fixed.yaml

