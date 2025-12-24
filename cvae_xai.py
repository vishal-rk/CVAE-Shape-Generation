cat > cvae_xai.py << 'EOF'
import os
import torch
import torch.utils.data
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from tqdm import tqdm

# Import model and dataset definitions from training script
# Make sure this import works - alternative is to copy the model class definition here
from cvae_training import CVAE, ServerDataset, one_hot, IMAGE_SIZE

# Output directories
BASE_DIR = 'cvae_demo'
CHECKPOINT_DIR = f'{BASE_DIR}/cvae_512_checkpoints'
XAI_DIR = f'{BASE_DIR}/cvae_512_xai'

# Create XAI directories
os.makedirs(XAI_DIR, exist_ok=True)
os.makedirs(f'{XAI_DIR}/latent_traversals', exist_ok=True)
os.makedirs(f'{XAI_DIR}/error_heatmaps', exist_ok=True)

# XAI Functions

def create_latent_traversals(model, epoch, class_idx, device, latent_size, n_steps=8, n_dims=10, traversal_range=(-5, 5)):
    """
    Create and save latent space traversals for specific dimensions.
    
    Args:
        model: Trained CVAE model
        epoch: Current epoch number
        class_idx: Class index to condition on
        device: Device to run the model on
        latent_size: Size of the latent space
        n_steps: Number of steps in the traversal
        n_dims: Number of dimensions to traverse (defaults to all dimensions)
        traversal_range: Range of values to traverse
    """
    # Default to traversing all latent dimensions if n_dims is None
    if n_dims is None:
        n_dims = latent_size
    
    # Create directory for this epoch's traversals
    epoch_dir = os.path.join(XAI_DIR, 'latent_traversals', f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    model.eval()  # Set model to evaluation mode
    
    # Create one-hot encoding for the class
    c = torch.zeros(1, model.class_size, device=device)
    c[0, class_idx] = 1.0
    
    # Create a base latent vector (zeros)
    z_base = torch.zeros(1, latent_size, device=device)
    
    # Generate traversal values
    traversal_values = torch.linspace(traversal_range[0], traversal_range[1], n_steps)
    
    # For each dimension, create a traversal
    with torch.no_grad():
        for dim_idx in range(min(n_dims, latent_size)):
            # Create empty tensor to store all steps of the traversal
            traversal_images = []
            
            for value in traversal_values:
                # Create a copy of the base latent vector
                z = z_base.clone()
                # Modify only the target dimension
                z[0, dim_idx] = value
                
                # Generate image from this latent vector
                generated = model.decode(z, c)
                generated = torch.clamp(generated, 0.0, 1.0)
                traversal_images.append(generated)
            
            # Concatenate all images in this traversal
            traversal_tensor = torch.cat(traversal_images, dim=0)
            
            # Save the traversal as an image grid
            save_image(
                traversal_tensor.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE),
                os.path.join(epoch_dir, f'traversal_class{class_idx}_dim{dim_idx}.png'),
                nrow=n_steps
            )
    
    print(f"Saved latent traversals for class {class_idx}, epoch {epoch} to {epoch_dir}")
    return epoch_dir

def create_reconstruction_error_heatmaps(model, epoch, data_loader, device, n_samples_per_class=1):
    """
    Create heatmaps showing reconstruction errors for one sample per class.
    
    Args:
        model: Trained CVAE model
        epoch: Current epoch number
        data_loader: DataLoader containing samples to visualize
        device: Device to run the model on
        n_samples_per_class: Number of samples per class to create heatmaps for
    """
    # Create directory for this epoch's error heatmaps
    epoch_dir = os.path.join(XAI_DIR, 'error_heatmaps', f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    model.eval()  # Set model to evaluation mode
    
    # Track how many samples we've processed per class
    samples_per_class = {c: 0 for c in range(model.class_size)}
    
    # Create a custom colormap - red for high error, blue for low error
    error_cmap = LinearSegmentedColormap.from_list('error_cmap', ['black', 'red', 'yellow', 'white'])
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            for i in range(len(data)):
                label_int = labels[i].item()
                if samples_per_class[label_int] < n_samples_per_class:
                    img = data[i:i+1].to(device)
                    label = labels[i:i+1].to(device)
                    label_onehot = torch.zeros(1, model.class_size, device=device)
                    label_onehot[0, label] = 1.0
                    
                    # Get reconstruction
                    recon, _, _ = model(img, label_onehot)
                    
                    # Calculate pixel-wise error (squared difference)
                    error = (recon - img).pow(2).squeeze().cpu().numpy()
                    
                    # Create the visualization
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axs[0].imshow(img.squeeze().cpu().numpy(), cmap='gray')
                    axs[0].set_title('Original')
                    axs[0].axis('off')
                    
                    axs[1].imshow(recon.squeeze().cpu().numpy(), cmap='gray')
                    axs[1].set_title('Reconstruction')
                    axs[1].axis('off')
                    
                    im = axs[2].imshow(error, cmap=error_cmap)
                    axs[2].set_title('Reconstruction Error')
                    axs[2].axis('off')
                    
                    cbar = fig.colorbar(im, ax=axs[2])
                    cbar.set_label('Squared Error')
                    
                    fig.suptitle(f'Epoch {epoch}, Class {label_int}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_dir, f'error_heatmap_class{label_int}_sample{samples_per_class[label_int]}.png'))
                    plt.close()
                    
                    samples_per_class[label_int] += 1
            
            # Stop if we've collected enough samples for all classes
            if all([samples_per_class[c] >= n_samples_per_class for c in samples_per_class]):
                break
    
    print(f"Saved {sum(samples_per_class.values())} error heatmap visualizations for epoch {epoch} to {epoch_dir}")
    return epoch_dir

def run_xai_for_epoch(epoch, model, val_loader, device, latent_size, class_names):
    """Run all XAI visualizations for a specific epoch"""
    print(f"\n{'='*60}")
    print(f"Generating XAI visualizations for EPOCH {epoch}")
    print(f"{'='*60}")
    
    # Clear CUDA cache before visualization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. Latent traversals for each class
    for class_idx in range(len(class_names)):
        create_latent_traversals(
            model=model,
            epoch=epoch,
            class_idx=class_idx,
            device=device,
            latent_size=latent_size,
            n_steps=8,              # Increased from 8 to 16 for smoother transitions
            n_dims=latent_size,      # Visualize all latent dimensions
            traversal_range=(-5, 5)  # Wider range to reveal subtle effects
        )
    
    # 2. Reconstruction error heatmaps
    create_reconstruction_error_heatmaps(
        model=model,
        epoch=epoch,
        data_loader=val_loader,
        device=device,
        n_samples_per_class=1        # One sample per class
    )
    
    # Clear CUDA cache after visualization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Completed XAI visualizations for epoch {epoch}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration - should match the training configuration
    latent_size = 100
    num_epochs = 30  # Total number of epochs to visualize
    batch_size = 8
    
    # Dataset configuration
    root_data_dir = "/pitsec_sose2025_team3_2/data"
    class_names = [
        'sized_squares_filled',
        'sized_squares_unfilled',
        'sized_rectangles_filled',
        'sized_rectangles_unfilled'
    ]
    num_classes = len(class_names)
    
    # Create validation dataset for error heatmaps
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    val_dataset = ServerDataset(root_dir=root_data_dir, class_names=class_names, 
                              split='val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = CVAE(latent_size=latent_size, class_size=num_classes).to(device)
    
    # Process each epoch
    for epoch in range(1, num_epochs + 1):
        # Load the checkpoint for this epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'cvae_model_epoch_{epoch}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for epoch {epoch} not found, skipping.")
            continue
        
        # Load the model weights
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Run XAI for this epoch
        run_xai_for_epoch(epoch, model, val_loader, device, latent_size, class_names)
    
    print("\nXAI visualization processing complete!")
    print(f"Visualizations saved to: {XAI_DIR}")

if __name__ == "__main__":
    main()
EOF