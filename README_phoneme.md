
# DrawSpeech — Phoneme Level Sketch Conditioning

## Overview

This project extends DrawSpeech (ICASSP 2025) by replacing word-level sketch conditioning with phoneme-level sketch conditioning, enabling finer-grained prosody control over synthesized speech.

**Original Paper:** DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions

**Our Contribution:** Instead of assigning one sketch value per word (which gets stretched to all phonemes in that word), we assign one sketch value per phoneme directly. This gives each individual sound its own pitch/energy guidance.

---

## The Key Change

### Original Paper (Word Level)

```
User draws sketch per word:
"didn't" = 0.8

F.interpolate stretches to phoneme level:
D=0.8  IH1=0.8  D=0.8  AH0=0.8  N=0.8  T=0.8

All phonemes in word get same value.
Imprecise.
```

### Our Change (Phoneme Level)

```
Sketch provided per phoneme directly:
D=0.1  IH1=0.9  D=0.1  AH0=0.6  N=0.2  T=0.1

F.pad just adds zeros to reach fixed length.
Each phoneme keeps its own individual value.
More precise.
```

### Code Change in dataset_plugin.py

**Before:**
```python
pitch_sketch = torch.from_numpy(pitch_sketch).float()[None, None, :]
pitch_sketch = F.interpolate(pitch_sketch,
               size=pitch_pad_length,
               mode="linear",
               align_corners=True).squeeze(0).squeeze(0)
```

**After:**
```python
pitch_sketch = sketch_extractor(pitch_sketch)
pitch_sketch = torch.from_numpy(pitch_sketch).float()
pitch_sketch = F.pad(pitch_sketch,
               (0, pitch_pad_length - pitch_sketch.size(0)),
               value=0)
pitch_sketch = min_max_normalize(pitch_sketch)
```

---

## Setup

### Requirements

```bash
conda create -n drawspeech python=3.10
conda activate drawspeech
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
pip install pytorch-lightning==1.9.5
pip install librosa pyworld tgt g2p_en soundfile
pip install pandas scikit-learn requests matplotlib Pillow
pip install inflect unidecode tqdm wandb pyyaml numpy==1.24.3
```

### Install taming-transformers

```bash
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
```

### Set PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/DrawSpeech_PyTorch:/path/to/taming-transformers
```

---

## Data Preparation

### Download LJSpeech

```bash
cd data/dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

### Download Alignments

Download LJSpeech.zip from FastSpeech2 repository (ming024/FastSpeech2) and extract:

```bash
cd data/dataset/LJSpeech-1.1
unzip LJSpeech.zip
```

### Download Checkpoints

Download from HuggingFace HappyColor/DrawSpeech:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='HappyColor/DrawSpeech', filename='vae.ckpt', local_dir='data/checkpoints')
"
```

### Run Preprocessing

```bash
python preprocessing.py
```

---

## Training

### From Scratch (Recommended)

```bash
sbatch train_scratch.sh
```

Or manually:

```bash
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py \
-c drawspeech/config/drawspeech_ljspeech_scratch.yaml
```

### With Fixed Pretrained Checkpoint

First fix the key names in the pretrained checkpoint:

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
print('Done')
"
```

Then train:

```bash
sbatch train_fixed.sh
```

---

## Inference

### With Phoneme Level Sketch

```bash
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py \
--config_yaml drawspeech/config/drawspeech_ljspeech_scratch.yaml \
--list_inference tests/inference_phoneme.json \
--reload_from_ckpt path/to/checkpoint.ckpt
```

### Create Custom Phoneme Sketch

```python
import numpy as np

# One value per phoneme
# Sentence: "I didn't say you stole the money"
# Phonemes: AY1 D IH1 D AH0 N T S EY1 Y UW1 S T OW1 L DH AH0 M AH1 N IY0

sketch = np.array([
    0.3,                              # AY1 = I
    0.1, 0.9, 0.1, 0.6, 0.2, 0.1,   # D IH1 D AH0 N T = didn't
    0.2, 0.7,                         # S EY1 = say
    0.3, 0.8,                         # Y UW1 = you
    0.5, 0.7, 0.9, 0.6,              # S T OW1 L = stole
    0.1, 0.1,                         # DH AH0 = the
    0.2, 0.3, 0.2                     # M AH1 N IY0 = money
])

np.save('my_sketch.npy', sketch)
```

---

## Results

| System | Pitch RMSE (Hz) | Training Steps |
|--------|----------------|----------------|
| Random sketch | 146.21 | - |
| Original mismatched checkpoint | 121.24 | 49k |
| Scratch training | 115.74 | 14k |
| Scratch training | 118.65 | 57k |
| Paper target | 62.48 | 80k |

### Key Finding

Scratch training at 14k steps (115.74 Hz) already outperforms original mismatched training at 49k steps (121.24 Hz). This confirms that proper initialization is critical for effective phoneme-level sketch conditioning.

---

## Files Changed From Original

| File | Change |
|------|--------|
| drawspeech/dataset_plugin.py | Replaced F.interpolate with F.pad for phoneme level sketch |
| drawspeech/infer.py | Fixed checkpoint loading, removed CLAP check |
| drawspeech/modules/fastspeech2/modules.py | Fixed mel_mask None issue |
| drawspeech/modules/latent_diffusion/ddpm.py | Fixed CLAP undefined error |
| drawspeech/utilities/audio/stft.py | Fixed pad_center argument for newer librosa |

---

## Citation

```
@INPROCEEDINGS{10889767,
  author={Chen, Weidong and Yang, Shan and Li, Guangzhi and Wu, Xixin},
  booktitle={ICASSP 2025},
  title={DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions},
  year={2025}
}
