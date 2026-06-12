

# DrawSpeech — Phoneme-Level Sketch Conditioning

This repository extends [DrawSpeech](https://arxiv.org/abs/2501.04256) (ICASSP 2025) by Chen et al. The original paper enables users to control speech prosody (pitch and energy) by drawing a sketch over the **words** of a sentence. This project extends that to **phoneme-level** sketch conditioning, giving finer-grained control over individual sounds.

**Original Paper:** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10889767) [[arXiv]](https://arxiv.org/abs/2501.04256) [[Demo]](https://happycolor.github.io/DrawSpeech/)
**Original Code:** [HappyColor/DrawSpeech_PyTorch](https://github.com/HappyColor/DrawSpeech_PyTorch)

---

## What's New in This Fork

```
1. Phoneme-level sketch conditioning
   F.interpolate (word → phoneme stretching) replaced with
   F.pad (direct phoneme-level input, no distortion)

2. Fixed checkpoint key-name mismatch
   Released checkpoint used old layer names
   (phoneme_encoder, detailed_curve_predictor)
   vs current code (text_encoder, sketch_to_contour_predictor)
   -> renamed keys, see drawspeech_fixed.ckpt creation below

3. Multiple training configs for comparison
   - drawspeech_ljspeech_22k.yaml   (original, mismatched ckpt)
   - drawspeech_ljspeech_scratch.yaml (scratch, VAE-only init)
   - drawspeech_ljspeech_fixed.yaml   (fixed checkpoint)
   - drawspeech_ljspeech_v2.yaml      (scratch + better logging/checkpointing)

4. Compatibility fixes for current PyTorch/librosa versions
   (stft.py, fastspeech2/modules.py, ddpm.py, infer.py)

5. Word-level vs phoneme-level sketch comparison tools
   (tests/word_sketches/, dataset_plugin_original.py)
```

---

# Getting Started

## 1. Environment Setup

```bash
conda create -n drawspeech python=3.10
conda activate drawspeech

pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
pip install pytorch-lightning==1.9.5
pip install librosa pyworld tgt g2p_en soundfile
pip install pandas scikit-learn requests matplotlib Pillow
pip install inflect unidecode tqdm wandb pyyaml numpy==1.24.3

git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers && pip install -e . && cd ..

export PYTHONPATH=$PYTHONPATH:/path/to/this/repo:/path/to/taming-transformers
```

---

## 2. Download Dataset and Checkpoints

### Dataset
Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) and place it as:

```plaintext
data/dataset/LJSpeech-1.1
 ┣ metadata.csv
 ┣ wavs/
 ┃ ┣ LJ001-0001.wav
 ┃ ┣ ...
 ┣ README
```

### Alignments
Download [LJSpeech.zip](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4) (TextGrid alignments from Montreal Forced Aligner, via FastSpeech2 repo) and unzip into `data/dataset/LJSpeech-1.1/`.

### Pretrained Checkpoints
Download from [HuggingFace HappyColor/DrawSpeech](https://huggingface.co/HappyColor/DrawSpeech/tree/main) and place into `data/checkpoints/`:

```
data/checkpoints/
 ┣ vae.ckpt              (Variational Autoencoder, frozen)
 ┣ drawspeech.ckpt       (original pretrained LDM, mismatched keys)
 ┗ LJ_V1/                (HiFi-GAN vocoder)
```

### Fix Checkpoint Key Names (Required for our configs)

The released `drawspeech.ckpt` has layer names from an older code version. Run:

```bash
python3 -c "
import torch
ckpt = torch.load('data/checkpoints/drawspeech.ckpt', map_location='cpu')
new_ckpt = {}
for k, v in ckpt.items():
    new_key = k.replace('phoneme_encoder', 'text_encoder')
    new_key = new_key.replace('detailed_curve_predictor', 'sketch_to_contour_predictor')
    new_ckpt[new_key] = v
torch.save(new_ckpt, 'data/checkpoints/drawspeech_fixed.ckpt')
print('Done. Old keys remaining:',
      sum('phoneme_encoder' in k or 'detailed_curve_predictor' in k for k in new_ckpt))
"
```

This produces `data/checkpoints/drawspeech_fixed.ckpt`, used by `drawspeech_ljspeech_fixed.yaml`.

---

## 3. Preprocessing

Extracts phoneme-level pitch, energy, and duration for all utterances using the TextGrid alignments:

```bash
python preprocessing.py
```

Output goes to `data/dataset/metadata/ljspeech/phoneme_level/` (pitch/, energy/, duration/, stats.json). This step is required before training or inference, since sketches are derived from these files.

---

# Training

All configs are in `drawspeech/config/`. The VAE and HiFi-GAN vocoder are always frozen (pretrained); only the Latent Diffusion Model (~94M params) is trained.

| Config | Initialization | Notes |
|---|---|---|
| `drawspeech_ljspeech_22k.yaml` | `drawspeech.ckpt` (strict=False) | Original — has key mismatch, some layers random |
| `drawspeech_ljspeech_scratch.yaml` | VAE only | All LDM weights from scratch |
| `drawspeech_ljspeech_fixed.yaml` | `drawspeech_fixed.ckpt` | Correctly loaded pretrained weights |
| `drawspeech_ljspeech_v2.yaml` | VAE only | Scratch + checkpoint every 2k steps, wandb online, val every 2 epochs |

### Run on SLURM (HPC)

```bash
sbatch scripts/train_scratch.sh   # or train_v2.sh, train_fixed.sh
```

### Run manually

```bash
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py \
  -c drawspeech/config/drawspeech_ljspeech_v2.yaml
```

Training resumes automatically from the latest checkpoint in `log/<config_name>/checkpoints/` if one exists.

---

# Inference

## A. Phoneme-Level Sketch (Our Contribution)

`tests/inference_phoneme.json` / `tests/inference_test_set.json` point `pitch_sketch` / `energy_sketch` directly at phoneme-level `.npy` files (one value per phoneme):

```bash
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py \
  --config_yaml drawspeech/config/drawspeech_ljspeech_scratch.yaml \
  --list_inference tests/inference_test_set.json \
  --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt
```

`dataset_plugin.py` (active by default) processes this with `sketch_extractor` (Savitzky-Golay smoothing) + `F.pad` — each phoneme keeps its own value.

## B. Word-Level Sketch (Original Paper Behaviour)

To reproduce the original word-level behaviour (`F.interpolate`), swap in the original plugin:

```bash
cp drawspeech/dataset_plugin_original.py drawspeech/dataset_plugin.py

CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py \
  --config_yaml drawspeech/config/drawspeech_ljspeech_scratch.yaml \
  --list_inference tests/inference_word_test_set.json \
  --reload_from_ckpt data/checkpoints/drawspeech_fixed.ckpt

# restore phoneme-level plugin afterwards
cp drawspeech/dataset_plugin_phoneme.py drawspeech/dataset_plugin.py
```

`tests/inference_word_test_set.json` uses `tests/word_sketches/word_sketch_*.npy` — one averaged value per word (created by grouping phonemes on `sp` boundaries).

## C. Custom Sketch

Create your own phoneme-level sketch (one float per phoneme in the transcription):

```python
import numpy as np

# Example: "I didn't say you stole the money"
# Phonemes: AY1 D IH1 D AH0 N T S EY1 Y UW1 S T OW1 L DH AH0 M AH1 N IY0
sketch = np.array([
    0.3,                              # AY1 = I
    0.1, 0.9, 0.1, 0.6, 0.2, 0.1,     # D IH1 D AH0 N T = didn't
    0.2, 0.7,                         # S EY1 = say
    0.3, 0.8,                         # Y UW1 = you
    0.5, 0.7, 0.9, 0.6,               # S T OW1 L = stole (emphasis)
    0.1, 0.1,                         # DH AH0 = the
    0.2, 0.3, 0.2                     # M AH1 N IY0 = money
])
np.save('my_sketch.npy', sketch)
```

Reference it in your inference JSON as `"pitch_sketch": "my_sketch.npy"`.

---

# Evaluation

We evaluate using **Pitch RMSE (Hz)**: extract pitch (librosa.yin) from generated and original audio, compare frame-by-frame.

```bash
python3 -c "
import numpy as np, librosa, os

test_files = ['LJ013-0220','LJ003-0040','LJ039-0136','LJ009-0239','LJ050-0082',
               'LJ011-0169','LJ042-0228','LJ006-0014','LJ028-0388','LJ012-0093']

rmse_values = []
for fname in test_files:
    orig, sr = librosa.load(f'data/dataset/LJSpeech-1.1/wavs/{fname}.wav', sr=22050)
    gen,  sr = librosa.load(f'path/to/generated/{fname}.wav', sr=22050)
    o_f0 = librosa.yin(orig, fmin=50, fmax=500)
    g_f0 = librosa.yin(gen,  fmin=50, fmax=500)
    n = min(len(o_f0), len(g_f0))
    rmse_values.append(np.sqrt(np.mean((o_f0[:n]-g_f0[:n])**2)))
print('Average RMSE:', np.mean(rmse_values).round(2), 'Hz')
"
```

To evaluate every saved checkpoint:

```bash
sbatch scripts/eval_all_checkpoints.sh
```

---

# Results

## Pitch RMSE Comparison (10 LJSpeech test sentences)

| System | Init | Steps | RMSE (Hz) |
|---|---|---|---|
| Random sketch | - | - | 146.21 |
| Mismatched `drawspeech.ckpt` (strict=False), phoneme sketch | mismatched | 0 | 122.14 |
| Mismatched, trained | mismatched | 49,000 | 121.24 |
| **Fixed checkpoint** (zero training), phoneme sketch | fixed | 0 | **116.07** |
| Scratch training | VAE only | 14,000 | 115.74 |
| Scratch training | VAE only | 80,000 | 120.98 |
| **v2 (scratch, better logging)** | VAE only | **2,000** | **115.27** |
| v2 | VAE only | 58,000 | ~122–126 |
| Paper-reported target | - | 80,000 | 62.48 |

**Key findings:**
- Renaming checkpoint keys alone (zero training) improves RMSE from 122.14 → 116.07 Hz — most of the "improvement" from training the mismatched checkpoint was actually just recovering from bad initialization.
- The main diffusion loss (`train/loss_simple`) converges within ~2,000 steps; the best test RMSE (115.27 Hz) is also reached at ~2,000 steps. Training beyond this does not improve — and slightly worsens — pitch RMSE, even though auxiliary pitch/energy predictor losses keep decreasing.
- We were unable to reproduce the paper's reported 62.48 Hz, even using the authors' exact checkpoint with corrected key names and zero additional training. The gap (~2x) is consistent across all configurations, suggesting a methodological difference in evaluation (test set, pitch-extraction settings, or normalization) rather than a training issue.

## Word-Level vs Phoneme-Level Sketch (single-sample, fixed checkpoint, zero training)

| Sketch Type | Sentence | RMSE (Hz) |
|---|---|---|
| Word-level (F.interpolate, 4 values → 135) | LJ013-0220 | 105.10 |
| Phoneme-level (F.pad, 58 values → 135) | LJ013-0220 | 130.72 |

This single-sample result does not support the hypothesis that phoneme-level conditioning lowers pitch RMSE — likely because word-level sketches (averaged per word) are smoother and easier for the model to track, while phoneme-level sketches demand more precise local control that pitch RMSE alone may not reward. A full 10-sample comparison and complementary metrics (e.g. sketch-to-output correlation, emphasis placement accuracy) are needed.

Figures for all experiments are in `results/figures/`.

---

# Repository Structure

```
.
├── drawspeech/
│   ├── dataset_plugin.py            # active plugin (phoneme-level, F.pad)
│   ├── dataset_plugin_phoneme.py    # backup of the above
│   ├── dataset_plugin_original.py   # original word-level plugin (F.interpolate)
│   ├── infer.py
│   ├── conditional_models.py
│   ├── config/
│   │   ├── drawspeech_ljspeech_22k.yaml
│   │   ├── drawspeech_ljspeech_scratch.yaml
│   │   ├── drawspeech_ljspeech_fixed.yaml
│   │   └── drawspeech_ljspeech_v2.yaml
│   ├── modules/
│   ├── train/
│   └── utilities/
├── tests/
│   ├── inference_phoneme.json
│   ├── inference_test_set.json
│   ├── inference_word_test_set.json
│   └── word_sketches/
├── scripts/
│   ├── train_scratch.sh
│   ├── train_v2.sh
│   ├── train_fixed.sh
│   └── eval_all_checkpoints.sh
├── results/
│   ├── figures/
│   └── v2_rmse_results.json
└── preprocessing.py
```

---

# Original Acknowledgements

This repository borrows code from:
* [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning)
* [FastSpeech 2](https://github.com/ming024/FastSpeech2)
* [HiFi-GAN](https://github.com/jik876/hifi-gan)

# Citation

```bibtex
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889767}}


