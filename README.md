# iDose-LT: Risk-Aware Reinforcement Learning for Tacrolimus Management After Liver Transplantation

This repository contains code to reproduce the **iDose-LT** framework and the figures for the manuscript.

> **Note**: The UHN dataset is not publicly shareable. The repository includes a synthetic data generator that mimics the structure (not the distribution) of the real data so that all pipelines and figures can be executed end-to-end for demonstration and CI testing.

## What’s here

- `src/idoselt/`
  - `data.py`: schema, synthetic data generator
  - `preprocess.py`: feature engineering and state construction
  - `model.py`: Risk-Aware DQN (PyTorch) with configurable risk terms
  - `train.py`: offline training loop with experience replay and target network
  - `eval.py`: evaluation utilities (KM survival, cumulative reward, risk trade-offs)
  - `plots.py`: figure generation (Fig. 2–7 equivalents on synthetic data)
- `configs/config.yaml`: training & evaluation config
- `scripts/`
  - `run_all.py`: end-to-end pipeline (generate synthetic data → train → evaluate → make figures)
  - `make_figures.py`: just remake plots from saved artifacts
- `figures/`: output plots
- `tests/`: minimal smoke tests
- `.github/workflows/ci.yaml`: basic CI (lint + quick run on synthetic data)

## Quick start

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
