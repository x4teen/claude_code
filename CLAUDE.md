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

## Models

| File | Dataset | Algorithm | Accuracy |
|---|---|---|---|
| `iris_classifier.py` | Iris (150 samples) | Random Forest (scikit-learn) | 90% |
| `mnist_cnn.py` | MNIST (70k images) | 2-layer CNN (PyTorch) | 99.1% |

MNIST downloads data to `data/` on first run (ignored by git). Trained weights (`*.pth`) are also gitignored.

## Notes

- PyTorch is CPU-only. To add GPU support, reinstall with the CUDA index URL from pytorch.org.
- `requirements.txt` is a full pinned freeze. When adding new packages, install then re-run `pip freeze > requirements.txt`.
