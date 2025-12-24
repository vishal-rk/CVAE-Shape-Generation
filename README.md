# CVAE-Shape-Generation

A Conditional Variational Autoencoder (CVAE) implementation for generating and analyzing geometric shapes with explainable AI (XAI) techniques.

## Overview

This project implements a deep learning model using Conditional Variational Autoencoders to generate 512×512 binary images of geometric shapes. The model supports four shape classes and includes comprehensive XAI analysis tools for understanding model behavior.

## Features

- **High-Resolution Generation**: 512×512 image generation with 6-layer convolutional architecture
- **Multi-Class Support**: 4 shape classes (filled/unfilled squares and rectangles)
- **Comprehensive Loss Tracking**: Per-batch and per-epoch tracking of BCE, KL divergence, MSE, and IoU metrics
- **XAI Techniques**:
  - t-SNE latent space visualization
  - DiCE counterfactual explanations
  - Confusion matrix analysis
  - Latent traversal visualization

## Architecture

- **Encoder**: 6-layer convolutional network with batch normalization
- **Latent Space**: 100-dimensional latent representation
- **Decoder**: 6-layer transposed convolutional network
- **Loss Components**: 
  - Binary Cross-Entropy (reconstruction)
  - KL Divergence (with annealing: 0.01 → 1.0)
  - Mean Squared Error
  - Intersection over Union (IoU)

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.

Main requirements:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Pillow
- tqdm

## Installation

```bash
# Clone the repository
git clone git@github.com:vishal-rk/CVAE-Shape-Generation.git
cd CVAE-Shape-Generation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

The model expects the following directory structure:

```
data/
└── [parent_dir]/
    ├── sized_squares_filled/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── sized_squares_unfilled/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── sized_rectangles_filled/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── sized_rectangles_unfilled/
        ├── train/
        ├── val/
        └── test/
```

## Usage

### Training

```bash
python cvae_training_cuda_fixed.py
```

**Training Configuration:**
- Epochs: 30
- Batch Size: 8
- Learning Rate: 5e-5
- Latent Size: 100
- Image Size: 512×512

The training script automatically:
- Saves model checkpoints every epoch
- Tracks and saves detailed loss metrics
- Generates sample images per epoch
- Creates reconstruction comparisons

**Output Directories:**
- `cvae_demo/cvae_512_checkpoints/`: Model checkpoints
- `cvae_demo/cvae_512_results/`: Generated samples and reconstructions
- `cvae_demo/cvae_512_losses/`: Comprehensive loss logs (per-batch and per-epoch)

### Testing & Evaluation

**Confusion Matrix:**
```bash
python cvae_confusion_matrix.py
```
Generates confusion matrix for reconstructed images based on IoU metrics.

**t-SNE Visualization:**
```bash
python cvae_xai.py
# or
python cvae_xai_100.py  # For 100-dimensional latent space
```
Creates latent space visualizations and traversal plots.

**DiCE Counterfactuals:**
```bash
python dice.py
# or
python cvae_counterfactual.py
```
Generates counterfactual explanations using multiple methods:
- Simple counterfactuals (direct class swap)
- Path interpolation
- Optimization-based counterfactuals
- Feature importance analysis

## Results

- **Best Model**: Epoch 26
- **Validation IoU**: ~0.9985
- **Architecture Stability**: KL annealing prevents posterior collapse
- **Latent Space**: Dimension 6 shows consistent geometric feature control

## Key Features

### Loss Tracking
The training script provides comprehensive loss tracking at multiple granularities:
- **Per-batch**: Individual batch losses saved as CSV and TXT files
- **Per-epoch**: Epoch-level summaries with all components
- **Complete history**: Full training history in JSON format

### XAI Analysis
Post-training analysis includes:
- Latent space exploration via t-SNE
- Counterfactual generation for understanding model decisions
- Confusion matrix for classification performance
- Latent dimension traversals showing learned features

## File Structure

```
├── cvae_training_cuda_fixed.py   # Main training script
├── cvae_confusion_matrix.py      # Confusion matrix generation
├── cvae_counterfactual.py        # Counterfactual analysis
├── cvae_xai.py                   # t-SNE and XAI visualizations
├── cvae_xai_100.py               # Extended XAI for 100D latent
├── dice.py                       # DiCE counterfactual implementation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Training Tips

1. **Memory Management**: Batch size of 8 is recommended for 512×512 images on typical GPUs
2. **KL Annealing**: Gradually increases from 0.01 to 1.0 to prevent posterior collapse
3. **Gradient Clipping**: Max norm of 1.0 prevents exploding gradients
4. **Checkpointing**: Model saves every epoch, allowing resumption from any point

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cvae-shape-generation,
  author = {Vishal RK},
  title = {CVAE-Shape-Generation: Conditional VAE for Geometric Shape Generation with XAI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/vishal-rk/CVAE-Shape-Generation}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with PyTorch
- XAI techniques inspired by DiCE and t-SNE literature
- Part of IT Security project (BPM)

## Contact

For questions or issues, please open an issue on GitHub or contact via the repository.
