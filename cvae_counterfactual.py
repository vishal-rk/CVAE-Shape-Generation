cat > cvae_counterfactual.py << 'EOF'
import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define constants
IMAGE_SIZE = 512

# CVAE model architecture - must match your training model
class CVAE(nn.Module):
    def __init__(self, latent_size, class_size):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.class_size = class_size
        self.image_size = IMAGE_SIZE

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + self.class_size, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.Conv2d(1024, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_size)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_size)

        # Decoder
        self.decoder_fc = nn.Linear(latent_size + class_size, 1024 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x, c):
        c = c.unsqueeze(-1).unsqueeze(-1).expand(-1, c.size(1), x.size(2), x.size(3))
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = self.decoder_fc(inputs).view(z.size(0), 1024, 8, 8)
        return self.decoder(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

# Dataset class for the test dataset structure
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split='test'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Define the class names (matching your training)
        self.class_names = [
            'sized_squares_filled',
            'sized_squares_unfilled',
            'sized_rectangles_filled',
            'sized_rectangles_unfilled'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Gather all image paths and their labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name, self.split)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        self.labels.append(class_idx)
            else:
                print(f"Warning: Directory not found: {class_dir}")
        
        print(f"Found {len(self.image_paths)} images in {self.split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('L')  # force grayscale
            if self.transform:
                img = self.transform(img)
            return img, label, os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            img = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return img, label, os.path.basename(img_path)

# Calculate IoU for binary shape masks
def calculate_iou(pred, target, threshold=0.5):
    # Binarize predictions and targets
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # Handle empty masks
    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    
    return intersection / union

# One-hot encoding helper
def one_hot(labels, class_size, device):
    targets = torch.zeros(labels.size(0), class_size, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return targets

def generate_counterfactuals(model, test_loader, device, num_classes, class_names, output_dir, max_samples=50):
    """
    Generate counterfactual reconstructions by encoding an image from one class
    and reconstructing it with different class conditions.
    """
    counterfactual_dir = os.path.join(output_dir, 'counterfactuals')
    os.makedirs(counterfactual_dir, exist_ok=True)
    
    # Create a file to store counterfactual metrics
    cf_metrics_file = os.path.join(output_dir, 'metrics', 'counterfactual_metrics.csv')
    with open(cf_metrics_file, 'w') as f:
        f.write("Filename,Original_Class,CF_Class,IoU,BCE_Loss,KL_Loss,MSE_Loss\n")
    
    # Track processed samples per class
    processed_per_class = {i: 0 for i in range(num_classes)}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels, filenames) in enumerate(tqdm(test_loader, desc="Generating Counterfactuals")):
            data = data.to(device)
            labels = labels.to(device)
            
            # Process each image
            for i, (img, label, filename) in enumerate(zip(data, labels, filenames)):
                # Skip if we've processed enough from this class
                if processed_per_class[label.item()] >= max_samples:
                    continue
                
                processed_per_class[label.item()] += 1
                original_class_idx = label.item()
                original_class_name = class_names[original_class_idx]
                
                # Get the latent representation with original class condition
                original_condition = torch.zeros(1, num_classes, device=device)
                original_condition[0, original_class_idx] = 1.0
                img_expanded = img.unsqueeze(0)  # Add batch dimension
                mu, logvar = model.encode(img_expanded, original_condition)
                z = model.reparameterize(mu, logvar)
                
                # Create a grid with original image and all counterfactual variations
                grid_images = [img_expanded.cpu()]
                
                # Create a figure for side-by-side comparison
                fig, axes = plt.subplots(1, num_classes + 1, figsize=(3*(num_classes + 1), 3))
                axes[0].imshow(img.cpu().squeeze().numpy(), cmap='gray')
                axes[0].set_title(f"Original\n({original_class_name})")
                axes[0].axis('off')
                
                # Generate counterfactuals for all classes
                for cf_class_idx in range(num_classes):
                    # Create counterfactual class condition
                    cf_condition = torch.zeros(1, num_classes, device=device)
                    cf_condition[0, cf_class_idx] = 1.0
                    
                    # Generate counterfactual reconstruction
                    cf_recon = model.decode(z, cf_condition)
                    grid_images.append(cf_recon.cpu())
                    
                    # Calculate metrics
                    iou = calculate_iou(cf_recon.cpu(), img_expanded.cpu())
                    bce_loss = F.binary_cross_entropy(
                        cf_recon.view(-1), img_expanded.view(-1), reduction='mean'
                    ).item()
                    mse_loss = F.mse_loss(
                        cf_recon.view(-1), img_expanded.view(-1), reduction='mean'
                    ).item()
                    
                    # Calculate KL divergence for reference
                    kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / 
                              mu.size(1)).item()
                    
                    # Save metrics
                    with open(cf_metrics_file, 'a') as f:
                        f.write(f"{filename},{original_class_name},{class_names[cf_class_idx]}," +
                                f"{iou:.6f},{bce_loss:.6f},{kl_loss:.6f},{mse_loss:.6f}\n")
                    
                    # Add to plot
                    axes[cf_class_idx + 1].imshow(cf_recon.cpu().squeeze().numpy(), cmap='gray')
                    axes[cf_class_idx + 1].set_title(f"As {class_names[cf_class_idx]}\nIoU: {iou:.4f}")
                    axes[cf_class_idx + 1].axis('off')
                
                # Save comparison plot
                plt.tight_layout()
                plt.savefig(os.path.join(counterfactual_dir, f"cf_{original_class_name}_{filename}.png"), 
                            dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save as image grid too
                save_image(torch.cat(grid_images), 
                          os.path.join(counterfactual_dir, f"cf_grid_{original_class_name}_{filename}"),
                          nrow=num_classes + 1, padding=5)
    
    print(f"Counterfactual images and metrics saved to {counterfactual_dir}")
    return cf_metrics_file

def analyze_counterfactuals(cf_metrics_file, output_dir, class_names):
    """
    Analyze counterfactual results and create visualizations
    """
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Read the metrics
    cf_metrics = pd.read_csv(cf_metrics_file)
    
    # Create a heatmap of average IoU for original class -> counterfactual class
    cf_pivot = cf_metrics.pivot_table(
        values='IoU', 
        index='Original_Class', 
        columns='CF_Class', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_pivot, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Counterfactual Reconstruction IoU\n(Original → Counterfactual)')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'counterfactual_heatmap.png'), dpi=300)
    plt.close()
    
    # Create a similar heatmap for BCE loss
    bce_pivot = cf_metrics.pivot_table(
        values='BCE_Loss', 
        index='Original_Class', 
        columns='CF_Class', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(bce_pivot, annot=True, fmt='.4f', cmap='rocket_r')
    plt.title('Counterfactual BCE Loss\n(Original → Counterfactual)')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'counterfactual_bce_heatmap.png'), dpi=300)
    plt.close()
    
    # Save summary to text file
    with open(os.path.join(analysis_dir, 'counterfactual_summary.txt'), 'w') as f:
        f.write("Counterfactual Analysis Summary\n")
        f.write("=============================\n\n")
        
        # Overall stats
        f.write(f"Overall average counterfactual IoU: {cf_metrics['IoU'].mean():.6f}\n")
        f.write(f"Best counterfactual transformation: {cf_metrics['IoU'].max():.6f}\n")
        f.write(f"Worst counterfactual transformation: {cf_metrics['IoU'].min():.6f}\n\n")
        
        # Class transformation stats
        f.write("Class transformation IoU matrix:\n")
        f.write(cf_pivot.to_string())
        f.write("\n\n")
        
        # Find most and least mutable classes
        class_mutability = {}
        for class_name in class_names:
            # Calculate average IoU when transforming FROM this class TO other classes
            from_iou = cf_metrics[cf_metrics['Original_Class'] == class_name]['IoU'].mean()
            class_mutability[class_name] = from_iou
        
        most_mutable = min(class_mutability.items(), key=lambda x: x[1])
        least_mutable = max(class_mutability.items(), key=lambda x: x[1])
        
        f.write(f"Most mutable class (easiest to transform): {most_mutable[0]} ({most_mutable[1]:.6f})\n")
        f.write(f"Least mutable class (hardest to transform): {least_mutable[0]} ({least_mutable[1]:.6f})\n")
    
    print(f"Counterfactual analysis saved to {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description='CVAE Counterfactual Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--data-dir', type=str, default='test_dataset',
                        help='Root directory of the test dataset')
    parser.add_argument('--output-dir', type=str, default='cvae_counterfactuals',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--latent-size', type=int, default=100,
                        help='Size of the latent space')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate (train, val, test)')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Maximum number of samples per class to process')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'analysis'), exist_ok=True)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    test_dataset = TestDataset(
        root_dir=args.data_dir,
        transform=transform,
        split=args.split
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize the model
    num_classes = len(test_dataset.class_names)
    model = CVAE(latent_size=args.latent_size, class_size=num_classes).to(device)
    
    # Load the model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate counterfactuals
    print("Generating counterfactual examples...")
    cf_metrics_file = generate_counterfactuals(
        model, 
        test_loader, 
        device, 
        num_classes, 
        test_dataset.class_names, 
        args.output_dir,
        max_samples=args.max_samples
    )
    
    # Analyze counterfactuals
    try:
        print("Analyzing counterfactual results...")
        analyze_counterfactuals(cf_metrics_file, args.output_dir, test_dataset.class_names)
    except Exception as e:
        print(f"Warning: Could not analyze counterfactual results: {e}")
        print("Make sure you have pandas and seaborn installed.")
    
    print(f"\nCounterfactual analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF