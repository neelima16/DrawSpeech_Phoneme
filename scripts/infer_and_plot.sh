#!/bin/bash
#SBATCH --job-name=infer_plot
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err
#SBATCH --cpus-per-task=4

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

echo "=== Starting Inference ==="

# Run inference on test set
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py \
--config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml \
--list_inference tests/inference_test_set.json \
--reload_from_ckpt "log/latent_diffusion/config/drawspeech_ljspeech_22k/checkpoints/checkpoint-fad-133.00-global_step=48999.ckpt"

echo "=== Inference Done ==="

# Calculate RMSE
python3 -c "
import numpy as np
import librosa
import os

infer_dir = 'log/latent_diffusion/config/drawspeech_ljspeech_22k'
folders = [f for f in os.listdir(infer_dir) if 'infer' in f]
folders.sort()
latest = os.path.join(infer_dir, folders[-1])
print('Using folder:', latest)

test_files = [
    'LJ013-0220', 'LJ003-0040', 'LJ039-0136', 'LJ009-0239', 'LJ050-0082',
    'LJ011-0169', 'LJ042-0228', 'LJ006-0014', 'LJ028-0388', 'LJ012-0093'
]

rmse_values = []
for fname in test_files:
    orig_path = f'data/dataset/LJSpeech-1.1/wavs/{fname}.wav'
    gen_path  = f'{latest}/{fname}.wav'
    if not os.path.exists(gen_path):
        print(f'Missing: {gen_path}')
        continue
    orig, sr = librosa.load(orig_path, sr=22050)
    gen, sr  = librosa.load(gen_path,  sr=22050)
    orig_f0 = librosa.yin(orig, fmin=50, fmax=500)
    gen_f0  = librosa.yin(gen,  fmin=50, fmax=500)
    min_len = min(len(orig_f0), len(gen_f0))
    rmse = np.sqrt(np.mean((orig_f0[:min_len] - gen_f0[:min_len])**2))
    rmse_values.append(rmse)
    print(f'{fname}: RMSE = {rmse:.2f} Hz')

print()
print('Average RMSE at 49k steps:', np.mean(rmse_values).round(2), 'Hz')
print('Previous 16k steps RMSE:   122.08 Hz')
print('Paper target:               62.48 Hz')
"

echo "=== Plotting Training Curves ==="

# Plot training curves
python3 -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0D1B2A')

# Training loss from checkpoints
train_steps  = [999,    1999,   2999,   5999,   15999,  48999]
train_losses = [0.2832, 0.2003, 0.1761, 0.1857, 0.1591, 0.198]

ax1.set_facecolor('#1a2a3a')
ax1.plot(train_steps, train_losses,
         color='#00B4D8', linewidth=2.5, marker='o', markersize=8)
ax1.set_xlabel('Training Steps', color='white', fontsize=12)
ax1.set_ylabel('Loss', color='white', fontsize=12)
ax1.set_title('Training Loss Curve\n(Phoneme Level Conditioning)',
              color='white', fontsize=13, fontweight='bold')
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for s, l in zip(train_steps, train_losses):
    ax1.annotate(str(l), (s, l), textcoords='offset points',
                xytext=(0, 10), color='#00B4D8', fontsize=8, ha='center')
ax1.axvline(x=80000, color='#FF9F1C', linestyle='--',
            linewidth=1.5, label='Paper target (80k)')
ax1.legend(facecolor='#0D1B2A', labelcolor='white')

# RMSE progress
rmse_steps  = [0,      6000,    16000,   49000]
rmse_values = [122.14, 111.83,  110.04,  0]
# placeholder for 49k, will update after measurement

ax2.set_facecolor('#1a2a3a')
ax2.plot(rmse_steps[:3], rmse_values[:3],
         color='#06D6A0', linewidth=2.5, marker='s', markersize=8)
ax2.axhline(y=62.48, color='#FF9F1C', linestyle='--',
            linewidth=1.5, label='Paper target (62.48 Hz)')
ax2.set_xlabel('Training Steps', color='white', fontsize=12)
ax2.set_ylabel('Pitch RMSE (Hz)', color='white', fontsize=12)
ax2.set_title('Validation RMSE Progress\n(Lower is Better)',
              color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(facecolor='#0D1B2A', labelcolor='white')
for s, r in zip(rmse_steps[:3], rmse_values[:3]):
    ax2.annotate(str(r), (s, r), textcoords='offset points',
                xytext=(0, 10), color='#06D6A0', fontsize=9, ha='center')
ax2.set_ylim([50, 160])

plt.suptitle('DrawSpeech Phoneme Level — Training Progress',
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_progress_49k.png', dpi=150,
            bbox_inches='tight', facecolor='#0D1B2A')
print('Saved training_progress_49k.png')
"

echo "=== All Done ==="
