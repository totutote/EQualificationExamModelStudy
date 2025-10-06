# Copilot Instructions for EQualificationExamModelStudy

## Project Overview

This repository contains implementations of various deep learning models and algorithms for E-Qualification exam study, focusing on PyTorch implementations. The project is organized into independent modules under `src/`, each implementing a specific ML/DL technique.

## Architecture & Organization

### Core Structure
- **`src/`**: Main implementation directory with independent ML modules
- **`dataset/`**: Shared datasets (CIFAR-10, FashionMNIST, VOC2012)
- **`outputs/`**: Generated results and visualizations
- **Pre-trained models**: Saved as `.pth` files in root directory

### Module Categories
1. **Computer Vision**: `dcgan/`, `conditional-gan/`, `denoising-autoencoder/`, `vae/`, `r-cnn/`, `ssd/`
2. **Reinforcement Learning**: `a3c-pendulum/`, `dqn-cartpole/`
3. **Natural Language Processing**: `seq2seq_attention.py`, `transformer/`
4. **Foundational**: `numpy/` (NumPy-based implementations from scratch)

## Development Patterns

### Model Implementation Structure
Each module follows a consistent pattern:
```
module-name/
├── utils.py          # Model definitions and utility functions
├── train.py          # Training loop implementation
├── predict.py        # Inference and visualization
└── README.md         # Module-specific documentation
```

### Device Configuration Pattern
Use this standard device detection across all modules:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```

### Model Naming Conventions
- **Generators**: `Generator` class in `utils.py`
- **Discriminators**: `Discriminator` class in `utils.py`
- **Agents**: `{Algorithm}Agent` (e.g., `DQNAgent`)
- **Core Models**: `{Algorithm}Model` (e.g., `A3CModel`)

### Data Loading Standards
- Use `torchvision.datasets` for standard datasets
- Implement custom transforms in training files
- Store datasets in `./dataset/` directory
- Batch size typically 64 for most experiments

## Key Implementation Details

### Output Management
- Save generated images to `./outputs/{module-name}/`
- Use incremental naming: `comparison_{i}.png`, `fake_images_step_{i}.png`
- Create output directories programmatically if they don't exist

### Model Persistence
- Save models as `.pth` files using `torch.save(model.state_dict(), path)`
- Load with `weights_only=True` parameter for security
- Store pre-trained models in project root

### Training Patterns
- Use `tqdm` for progress bars in training loops
- Implement validation loops in training scripts
- Print loss information each epoch
- Support both GPU (CUDA) and Apple Silicon (MPS)

## Environment Setup

### Dependencies
- Primary environment: `environment.yml` (conda)
- Windows variant: `environment_win.yml`
- Docker support via `Dockerfile`
- Core dependencies: PyTorch 2.5.1, torchvision, gymnasium

### Device Testing
Run `src/environment_test.py` to verify CUDA/MPS availability before training.

## Module-Specific Notes

### GANs (DCGAN, Conditional-GAN)
- Use `DEFAULT_CHANNELS = 128` as standard channel count
- Implement weight initialization with `waight_init` function
- Discriminator and Generator learning rates often differ

### Reinforcement Learning
- Environment videos saved to `video/` directory
- Support for both discrete (CartPole) and continuous (Pendulum) action spaces
- Use gymnasium environments with video recording capabilities

### AutoEncoders
- Implement noise addition utilities for denoising variants
- Use sigmoid activation for output layers
- Support comparison visualizations (original/noisy/denoised)

## Common Utilities

### NumPy Implementations
The `numpy/` directory contains educational from-scratch implementations of:
- Affine transformations with backward propagation
- Activation functions (ReLU, Sigmoid)
- Basic neural network building blocks

Use these as reference for understanding underlying mathematics before implementing PyTorch versions.