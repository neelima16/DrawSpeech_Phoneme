#!/bin/bash
#SBATCH --job-name=draw_v2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/v2_%j.log
#SBATCH --error=logs/v2_%j.err
#SBATCH --cpus-per-task=8

# Load modules
module load python
conda activate drawspeech

# Set proxy
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80

# Login to wandb
export WANDB_API_KEY=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
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=300

# Set paths
export PYTHONPATH=$PYTHONPATH:/home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch:/home/hpc/iwi5/iwi5408h/taming-transformers

# Go to project directory
cd /home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch

# Train v2
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py \
-c drawspeech/config/drawspeech_ljspeech_v2.yaml

