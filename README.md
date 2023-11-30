# AAI3001 Large Project
This repository serves as an archive for a semantic segmentation task on the MoNuSeg Dataset.

# Requirements
- A CUDA enabled GPU.
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Python >= 3.8

# Installation
Run either
```bash
pip install -r requirements.txt
```
or
```bash
pipenv install
```

# Usage
## Training
Edit the constants at the top of `train.py` and run:
```bash
python train.py
```

## Evaluation 1 - Classification Metrics
Edit the constants at the top of `validation.py` and run:
```bash
python validation.py
```

## Evaluation 2 - Adversarial Attacks
### White box - Projected Gradient Descent
Run
```bash
python whitebox_pgd.py
python whitebox_pgd_eval.py
```

### White box - Basic Iterative Method
Run
```bash
python whiteboxBIM.py
python whiteboxBIM_eval.py
```

### White box - Fast Gradient Sign Method
Run
```bash
python whitebox_fgsm.py
python whitebox_fgsm_eval.py
```

# Licence
[BSD-3 Clause](LICENSE.txt)