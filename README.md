# Scene Recognition & Semantic Segmentation (Deep Learning)

This repository contains two self-contained projects:

- Scene Recognition with convolutional networks and ResNet variants
- Semantic Segmentation on CamVid and KITTI with simple networks and PSPNet

Both projects include Jupyter notebooks, PyTorch training utilities, and conda environments for reproducibility on macOS.

## Repository Layout

- Scene Recognition with Deep Learning/
	- proj3.ipynb — notebook with end-to-end training/evaluation
	- src/vision/ — PyTorch dataloaders, models (`SimpleNet`, `MyResNet18`, `MultilabelResNet18`), training loop
	- data/ — 15-class scene dataset folders with `train/` and `test/`
	- conda/environment.yml — conda environment spec (`cv_proj3`)
	- trained_*.pt — example trained weights
- Semantic Segmentation with Deep Learning/
	- proj4.ipynb, proj4_local.ipynb — notebooks with training/evaluation
	- src/vision/ — transforms, datasets, networks (SimpleSegmentationNet, PSPNet), IoU utilities
	- Camvid/, kitti/ — dataset locations and samples
	- download_dataset.sh — helper to fetch CamVid images
	- conda/environment.yml — conda environment spec (`cv_proj4`)

## Prerequisites

- macOS (Intel or Apple Silicon)
- Conda (miniconda or Anaconda)
- Python 3.8+ recommended (project specs allow `python>=3.6`)

## Setup

Create project-specific environments (recommended to use separate envs):

```bash
# Scene Recognition environment
conda env create -f "Scene Recognition with Deep Learning/conda/environment.yml"
conda activate cv_proj3

# Semantic Segmentation environment
conda env create -f "Semantic Segmentation with Deep Learning/conda/environment.yml"
conda activate cv_proj4
```

Notes:
- On Apple Silicon/macOS, PyTorch uses CPU or Metal Performance Shaders (MPS). CUDA wheels listed in `proj4` are for Linux/Windows; conda will resolve macOS-compatible builds automatically. If needed, install macOS builds via `pip install torch torchvision torchaudio`.

## Data

### Scene Recognition
Data is expected under:

- `Scene Recognition with Deep Learning/data/train/<class>/` and `.../test/<class>/`
- Classes are folders (alphabetically sorted) like `bedroom`, `coast`, `forest`, ...

### Semantic Segmentation

CamVid:

```bash
chmod +x "Semantic Segmentation with Deep Learning/download_dataset.sh"
./Semantic Segmentation with Deep Learning/download_dataset.sh "Semantic Segmentation with Deep Learning/Camvid"
```

KITTI: Place training/testing images and labels under `Semantic Segmentation with Deep Learning/kitti/` following the existing `training/` and `testing/` layout.

## Quickstart

### Scene Recognition (Notebook)

```bash
conda activate cv_proj3
jupyter lab
```

Open `Scene Recognition with Deep Learning/proj3.ipynb` and run the cells to train and evaluate `SimpleNet`/`MyResNet18`. The notebook uses utilities in `src/vision/`:

- Dataloaders: `ImageLoader`, `MultiLabelImageLoader`
- Transforms: `data_transforms.py`
- Training loop: `runner.py` (`Trainer`, `MultiLabelTrainer`)

### Semantic Segmentation (Notebook)

```bash
conda activate cv_proj4
jupyter lab
```

Open `Semantic Segmentation with Deep Learning/proj4.ipynb` (or `proj4_local.ipynb`) and run the cells. The notebooks cover:

- Preparing dataloaders and transforms (`src/vision/cv2_transforms.py`, dataset lists)
- Training simple segmentation networks and PSPNet
- Evaluation with IoU/accuracy (`src/vision/iou.py`, `accuracy_calculator.py`)

## Pretrained Weights

Example trained weights are provided under both project folders (`trained_*.pt`). Use the notebooks to load and evaluate these weights. If using `runner.py`, the trainer saves/loads checkpoints from a `checkpoint.pt` inside `model_dir`.

## macOS Tips

- Apple Silicon: set `mps=True` in `Trainer` to enable GPU acceleration via Metal.
- Intel macOS: use CPU (`cuda=False, mps=False`).
- If Jupyter shows the wrong kernel, select the conda env kernel from the kernel picker or run: `python -m ipykernel install --user --name cv_proj3` and similarly for `cv_proj4`.

## Testing & Linting

Both environments include `pytest`, `flake8`, `mypy`, and `bandit`.

```bash
# Example (run from project root)
conda activate cv_proj3
pytest
flake8
mypy Scene\ Recognition\ with\ Deep\ Learning/src
bandit -r Scene\ Recognition\ with\ Deep\ Learning/src
```

