# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Public ML/data science research projects, driven primarily by Jupyter notebooks. The `helper_functions/` directory is a shared Python package installed locally and imported across notebooks.

## Setup

```bash
pip install -e .          # install helper_functions as an editable package
pip install -r requirements.txt
jupyter notebook
```

## Package Structure (`helper_functions/`)

- `general/` — feature generation, feature selection, model selection, training helpers
- `afml_functions/` — utilities from Advances in Financial Machine Learning (fracDiff, data analysis, modelling)
- `volatility_prediction_functions/` — volatility-specific helpers

## Research Projects (`research_projects/`)

- `football_prediction.ipynb` / `football_prediction_NN.ipynb` — match outcome prediction
- `volatility_prediction_optiver.ipynb` / `volatility_entropy_pred.ipynb` — volatility forecasting
- `fx_usdsek_projext.ipynb` — FX price modelling with fractional differentiation and rolling features
- `RL-Cartpole/` — DQN reinforcement learning on CartPole (modular: `Algo.py`, `Env.py`, `Model.py`, `Replay.py`)

## Git / SSH

`HOME` is set to the `dev/` folder (via `.bashrc`), so SSH looks for keys in `~/OneDrive/Documents/dev/.ssh/` rather than the real user profile. An SSH config at `~/OneDrive/Documents/dev/.ssh/config` redirects this to the actual key and known_hosts:

```
Host *
  IdentityFile /c/Users/camer/.ssh/id_ed25519
  UserKnownHostsFile /c/Users/camer/.ssh/known_hosts
```

## Stack

numpy, pandas, scikit-learn, statsmodels, scipy, matplotlib, seaborn, torch, gym
