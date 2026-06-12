#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_v2_%j.log
#SBATCH --error=logs/eval_v2_%j.err
#SBATCH --cpus-per-task=4

module load python
conda activate drawspeech

export PYTHONPATH=$PYTHONPATH:/home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch:/home/hpc/iwi5/iwi5408h/taming-transformers

cd /home/hpc/iwi5/iwi5408h/DrawSpeech_PyTorch

CKPT_DIR="log/latent_diffusion_v2/config/drawspeech_ljspeech_v2/checkpoints"

for ckpt in "$CKPT_DIR"/checkpoint-fad-*.ckpt; do
    echo "=== Running inference for $ckpt ==="
    CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py \
    --config_yaml drawspeech/config/drawspeech_ljspeech_v2.yaml \
    --list_inference tests/inference_test_set.json \
    --reload_from_ckpt "$ckpt"
done

echo "=== All Done ==="
