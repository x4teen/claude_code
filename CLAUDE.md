# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Test/learning repository for exploring Claude Code. Uses Python 3.12 with a deep learning stack.

## Environment

The virtual environment is in `.venv/`. Activate it before running anything:

```bash
source .venv/Scripts/activate  # Git Bash
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Key Libraries

- **PyTorch 2.10 (CPU)** — `torch`, `torchvision`, `torchaudio`
- **Data** — `numpy`, `pandas`, `scipy`
- **Visualization** — `matplotlib`
- **Classical ML** — `scikit-learn`
- **Notebooks** — `jupyterlab`, `ipykernel`
- **Utilities** — `tqdm`

## Running JupyterLab

```bash
jupyter lab
```

## Notes

- PyTorch is CPU-only. To add GPU support, reinstall with the CUDA index URL from pytorch.org.
- `requirements.txt` is a full pinned freeze. When adding new packages, install then re-run `pip freeze > requirements.txt`.
