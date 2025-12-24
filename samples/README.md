# Sample Output Images

This directory contains example outputs from the CVAE model to showcase its capabilities.

## What to Include

Add a few representative images here, such as:

1. **Generated Samples** (`generated_samples.png`)
   - One image per class showing the model's generation capabilities
   - From: `cvae_demo/cvae_512_results/sample_epoch_XX.png`

2. **Reconstruction Examples** (`reconstructions.png`)
   - Side-by-side comparison of original vs reconstructed images
   - From: `cvae_demo/cvae_512_results/reconstruction_epoch_XX.png`

3. **Latent Traversal** (`latent_traversal.png`)
   - Visualization showing smooth transitions in latent space
   - From: `latent_traversals/` or `enhanced_traversals/`

4. **Confusion Matrix** (`confusion_matrix.png`)
   - Classification performance visualization
   - Generated from `cvae_confusion_matrix.py`

## How to Add Images

```bash
# Copy your best results here
cp cvae_demo/cvae_512_results/sample_epoch_26.png samples/generated_samples.png
cp cvae_demo/cvae_512_results/reconstruction_epoch_26.png samples/reconstructions.png
```

After adding images, update the main README.md to reference them for better project visibility.
