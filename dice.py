cat > cvae_xai.py << 'EOF'import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define constants
IMAGE_SIZE = 512

# CVAE model architecture
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

# Dataset class for the test dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split='test'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Define the class names
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
        
        exts = {'.bmp', '.png', '.jpg', '.jpeg', '.gif'}
        
        # Path format - check both potential directory structures
        for parent_dir in os.listdir(root_dir):
            parent_path = os.path.join(root_dir, parent_dir)
            if not os.path.isdir(parent_path):
                continue
                
            # Check subdirectories for class folders
            for class_name in os.listdir(parent_path):
                if class_name in self.class_names:
                    class_idx = self.class_to_idx[class_name]
                    split_dir = os.path.join(parent_path, class_name, self.split)
                    
                    if not os.path.isdir(split_dir):
                        # Try alternative structure with class name directly in root_dir
                        split_dir = os.path.join(root_dir, class_name, self.split)
                        if not os.path.isdir(split_dir):
                            print(f"Warning: Directory not found, skipping: {split_dir}")
                            continue
                    
                    # Get all image files in that directory
                    for fname in sorted(os.listdir(split_dir)):
                        if os.path.splitext(fname.lower())[1] in exts:
                            path = os.path.join(split_dir, fname)
                            self.image_paths.append(path)
                            self.labels.append(class_idx)
        
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
            # Ensure pixel values are in valid range [0,1]
            img = torch.clamp(img, 0.0, 1.0)
            return img, label, os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
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

# DiCE counterfactual generation
class DiceCVAE:
    """
    DiCE-inspired counterfactual explanation generator for CVAE models
    """
    def __init__(self, model, device, num_classes, class_names, output_dir):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.output_dir = output_dir
        self.dice_dir = os.path.join(output_dir, 'dice_counterfactuals')
        os.makedirs(self.dice_dir, exist_ok=True)
        
    def generate_counterfactuals(self, img, true_class, target_class, filename, skip_optimization=False):
        """Generate counterfactuals for a given image from true class to target class"""
        # Prepare image and get latent encoding
        img_expanded = img.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Create true class condition
        true_condition = torch.zeros(1, self.num_classes, device=self.device)
        true_condition[0, true_class] = 1.0
        
        # Create target class condition
        target_condition = torch.zeros(1, self.num_classes, device=self.device)
        target_condition[0, target_class] = 1.0
        
        with torch.no_grad():
            # Encode the input image
            mu, logvar = self.model.encode(img_expanded, true_condition)
            z = self.model.reparameterize(mu, logvar)
            
            # Reconstruct with original class
            original_recon = self.model.decode(z, true_condition)
            
            # Method 1: Simple target class swap
            target_recon = self.model.decode(z, target_condition)
            
        # Save simple counterfactual comparison
        self._save_simple_counterfactual(
            img_expanded, original_recon, target_recon, 
            true_class, target_class, filename
        )
        
        # Method 2: Path interpolation
        self._generate_path_counterfactual(
            z, true_condition, target_condition, 
            img_expanded, true_class, target_class, filename
        )
        
        # Method 3: Latent optimization - only if not skipped
        if not skip_optimization:
            try:
                self._generate_optimized_counterfactual(
                    z, true_condition, target_condition,
                    img_expanded, true_class, target_class, filename
                )
            except Exception as e:
                print(f"Optimization error for {filename}: {e}")
        
        # Method 4: Latent feature importance
        self._generate_feature_importance(
            z, true_condition, target_condition, 
            true_class, target_class, filename
        )
        
        return {
            'original': img_expanded.cpu(),
            'original_recon': original_recon.cpu(),
            'target_recon': target_recon.cpu()
        }
    
    def _save_simple_counterfactual(self, img, original_recon, target_recon, 
                                   true_class, target_class, filename):
        """Save simple counterfactual comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].set_title(f"Original\n({self.class_names[true_class]})")
        axes[0].axis('off')
        
        # Original reconstruction
        axes[1].imshow(original_recon.cpu().squeeze().numpy(), cmap='gray')
        axes[1].set_title(f"Reconstruction\n({self.class_names[true_class]})")
        axes[1].axis('off')
        
        # Target counterfactual
        axes[2].imshow(target_recon.cpu().squeeze().numpy(), cmap='gray')
        axes[2].set_title(f"Counterfactual\n({self.class_names[target_class]})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dice_dir, f"simple_cf_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()
    
    def _generate_path_counterfactual(self, z, true_condition, target_condition, 
                                     img, true_class, target_class, filename):
        """Generate path interpolation between classes"""
        with torch.no_grad():
            num_steps = 8
            fig, axes = plt.subplots(2, num_steps, figsize=(num_steps*2, 6))
            
            # Generate counterfactual path
            for step in range(num_steps):
                # Linear interpolation of class condition
                alpha = step / (num_steps - 1)
                interp_condition = (1 - alpha) * true_condition + alpha * target_condition
                
                # Generate reconstruction with interpolated condition
                interp_recon = self.model.decode(z, interp_condition)
                
                # Calculate IOU with original image
                iou = calculate_iou(interp_recon.cpu(), img.cpu())
                
                # Plot reconstruction
                axes[0, step].imshow(interp_recon.cpu().squeeze().numpy(), cmap='gray')
                axes[0, step].set_title(f"Step {step+1}\nIoU: {iou:.4f}")
                axes[0, step].axis('off')
                
                # Plot class condition weights
                bars = axes[1, step].bar(range(self.num_classes), interp_condition.cpu().squeeze().numpy())
                for idx, bar in enumerate(bars):
                    if idx == true_class:
                        bar.set_color('blue')
                    elif idx == target_class:
                        bar.set_color('red')
                axes[1, step].set_ylim(0, 1)
                axes[1, step].set_title(f"Class Weights")
                axes[1, step].set_xticks(range(self.num_classes))
                axes[1, step].set_xticklabels([c[:1] for c in self.class_names], rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dice_dir, f"path_cf_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                       dpi=200, bbox_inches='tight')
            plt.close()
    
    def _generate_optimized_counterfactual(self, z, true_condition, target_condition,
                                         img, true_class, target_class, filename):
        """Generate optimized counterfactual by finding minimal latent space changes"""
        # Prepare target reconstruction for reference
        with torch.no_grad():
            target_recon = self.model.decode(z, target_condition)
            # Store the original z for reference (detached)
            z_original = z.clone().detach()
        
        # Create optimizable latent vector - CLONE AND DETACH to fix gradient issue
        z_cf = z.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_cf], lr=0.05)
        
        # Set up for visualization
        steps_to_save = 10
        save_interval = 20
        all_recons = []
        all_losses = []
        
        # Optimize for minimal change in latent space
        for step in range(steps_to_save * save_interval):
            # Generate reconstruction with current latent
            cf_recon = self.model.decode(z_cf, true_condition)
            
            # Calculate losses - FIXED: making sure we're computing against detached targets
            # 1. Reconstruction loss - want to look like the target class
            recon_loss = F.mse_loss(cf_recon, target_recon.detach())
            
            # 2. Latent loss - want minimal change in latent space
            latent_loss = F.mse_loss(z_cf, z_original)
            
            # Combined loss with weighting
            loss = recon_loss + 0.1 * latent_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()  # This should now work
            optimizer.step()
            
            # Record progress
            if step % save_interval == 0:
                all_recons.append(cf_recon.detach().cpu())
                all_losses.append(loss.item())
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].set_title(f"Original\n({self.class_names[true_class]})")
        axes[0].axis('off')
        
        # Target reconstruction
        axes[1].imshow(target_recon.detach().cpu().squeeze().numpy(), cmap='gray')
        axes[1].set_title(f"Target\n({self.class_names[target_class]})")
        axes[1].axis('off')
        
        # Optimization steps
        for i, recon in enumerate(all_recons):
            if i < 10:  # Make sure we don't exceed our subplot grid
                axes[i+2].imshow(recon.squeeze().numpy(), cmap='gray')
                axes[i+2].set_title(f"Step {(i+1)*save_interval}\nLoss: {all_losses[i]:.4f}")
                axes[i+2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dice_dir, f"opt_cf_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                   dpi=200, bbox_inches='tight')
        plt.close()
        
        # Create a plot showing loss over time
        plt.figure(figsize=(8, 4))
        plt.plot([(i+1)*save_interval for i in range(len(all_losses))], all_losses)
        plt.title('Counterfactual Optimization')
        plt.xlabel('Optimization Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dice_dir, f"opt_loss_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                   dpi=200)
        plt.close()
    
    def _generate_feature_importance(self, z, true_condition, target_condition,
                                   true_class, target_class, filename):
        """Generate feature importance for counterfactual explanation"""
        with torch.no_grad():
            # Original reconstruction
            original_recon = self.model.decode(z, true_condition)
            
            # Reference reconstruction with target class
            target_recon = self.model.decode(z, target_condition)
            
            # For each latent dimension, measure its importance
            latent_size = z.size(1)
            feature_importance = []
            perturbed_recons = []
            
            for dim in range(min(10, latent_size)):  # Analyze first 10 dimensions for visualization
                # Create a perturbed latent vector with one dimension changed
                z_perturbed = z.clone()
                
                # Modify this dimension
                z_perturbed[0, dim] = z[0, dim] + 2.0  # Significant perturbation
                
                # Generate reconstruction with perturbed latent
                perturbed_recon = self.model.decode(z_perturbed, true_condition)
                perturbed_recons.append(perturbed_recon.cpu())
                
                # Measure how much this affects the reconstruction (difference from original)
                diff = F.mse_loss(perturbed_recon, original_recon).item()
                feature_importance.append(diff)
        
        # Create visualization of feature importance
        feature_importance = np.array(feature_importance)
        if np.max(feature_importance) > 0:
            feature_importance = feature_importance / np.max(feature_importance)  # Normalize
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title(f'Latent Feature Importance for {self.class_names[true_class]} to {self.class_names[target_class]}')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Normalized Importance')
        plt.savefig(os.path.join(self.dice_dir, f"feature_imp_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                   dpi=200)
        plt.close()
        
        # Show effect of perturbing each dimension
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, recon in enumerate(perturbed_recons):
            if i < 10:  # Show first 10 dimensions
                axes[i].imshow(recon.squeeze().numpy(), cmap='gray')
                axes[i].set_title(f"Dim {i}: {feature_importance[i]:.4f}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dice_dir, f"feature_perturb_{self.class_names[true_class]}_to_{self.class_names[target_class]}_{filename}.png"),
                   dpi=200, bbox_inches='tight')
        plt.close()

def generate_all_counterfactuals(model, test_loader, device, num_classes, class_names, output_dir, max_samples=10, skip_optimization=False):
    """Generate diverse counterfactual explanations for images in the test loader"""
    dice_cvae = DiceCVAE(model, device, num_classes, class_names, output_dir)
    
    # Track processed samples per class
    processed_per_class = {i: 0 for i in range(num_classes)}
    
    # Summary data
    counterfactual_info = {
        (source, target): [] for source in range(num_classes) for target in range(num_classes) if source != target
    }
    
    model.eval()
    with torch.no_grad():
        for data, labels, filenames in tqdm(test_loader, desc="Generating DiCE counterfactuals"):
            data = data.to(device)
            labels = labels.to(device)
            
            # Process each image
            for i, (img, label, filename) in enumerate(zip(data, labels, filenames)):
                try:
                    true_class = label.item()
                    
                    # Skip if we've processed enough from this class
                    if processed_per_class[true_class] >= max_samples:
                        continue
                    
                    processed_per_class[true_class] += 1
                    
                    # Generate counterfactuals for all other classes
                    for target_class in range(num_classes):
                        if target_class == true_class:
                            continue
                        
                        # Generate counterfactuals
                        cf_results = dice_cvae.generate_counterfactuals(
                            img, true_class, target_class, filename, skip_optimization
                        )
                        
                        # Calculate metrics
                        original = cf_results['original']
                        target_recon = cf_results['target_recon']
                        iou = calculate_iou(target_recon, original)
                        
                        # Record in summary
                        counterfactual_info[(true_class, target_class)].append({
                            'filename': filename,
                            'iou': iou
                        })
                except Exception as e:
                    print(f"Error processing image {filenames[i] if i < len(filenames) else 'unknown'}: {e}")
                    continue
    
    # Create summary visualization for counterfactual quality
    cf_quality = np.zeros((num_classes, num_classes))
    for (source, target), items in counterfactual_info.items():
        if items:
            cf_quality[source, target] = np.mean([item['iou'] for item in items])
    
    # Mask the diagonal (no counterfactuals for same class)
    np.fill_diagonal(cf_quality, np.nan)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_quality, annot=True, fmt='.4f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names, 
                cbar_kws={'label': 'Average IoU'})
    plt.xlabel('Counterfactual Target Class')
    plt.ylabel('Original Class')
    plt.title('Counterfactual Quality (IoU)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_counterfactuals', 'counterfactual_quality.png'), dpi=300)
    plt.close()
    
    # Generate summary report
    with open(os.path.join(output_dir, 'dice_counterfactuals', 'counterfactual_report.txt'), 'w') as f:
        f.write("DiCE Counterfactual Analysis\n")
        f.write("===========================\n\n")
        
        # Overall stats
        all_ious = []
        for items in counterfactual_info.values():
            all_ious.extend([item['iou'] for item in items])
        
        if all_ious:
            f.write(f"Overall average counterfactual IoU: {np.mean(all_ious):.4f}\n\n")
        else:
            f.write("No valid counterfactuals generated.\n\n")
        
        # Per-transformation stats
        f.write("Counterfactual quality by class transformation:\n\n")
        for (source, target), items in counterfactual_info.items():
            if items:
                avg_iou = np.mean([item['iou'] for item in items])
                f.write(f"{class_names[source]} → {class_names[target]}: {avg_iou:.4f}\n")
        
        f.write("\n")
        
        # Most and least successful transformations
        if counterfactual_info:
            try:
                best_key = max(counterfactual_info.items(), 
                              key=lambda x: np.mean([item['iou'] for item in x[1]]) if x[1] else 0)
                worst_key = min(counterfactual_info.items(),
                               key=lambda x: np.mean([item['iou'] for item in x[1]]) if x[1] else float('inf'))
                
                if best_key[1]:
                    best_source, best_target = best_key[0]
                    best_avg = np.mean([item['iou'] for item in best_key[1]])
                    f.write(f"Most successful transformation: {class_names[best_source]} → {class_names[best_target]} ({best_avg:.4f})\n")
                
                if worst_key[1]:
                    worst_source, worst_target = worst_key[0]
                    worst_avg = np.mean([item['iou'] for item in worst_key[1]])
                    f.write(f"Least successful transformation: {class_names[worst_source]} → {class_names[worst_target]} ({worst_avg:.4f})\n")
            except Exception as e:
                f.write(f"Error calculating best/worst transformations: {e}\n")
    
    print(f"DiCE counterfactual analysis completed and saved to {os.path.join(output_dir, 'dice_counterfactuals')}")
    return counterfactual_info

def main():
    parser = argparse.ArgumentParser(description='CVAE DiCE Counterfactual Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--data-dir', type=str, default='test_dataset',
                        help='Root directory of the test dataset')
    parser.add_argument('--output-dir', type=str, default='cvae_dice_counterfactuals',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--latent-size', type=int, default=100,
                        help='Size of the latent space')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate (train, val, test)')
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Maximum number of samples per class to process')
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Skip the optimization-based counterfactual generation')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'dice_counterfactuals'), exist_ok=True)
    
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
        num_workers=1,
        pin_memory=False  # Disable pin_memory to reduce memory usage
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
    
    # Generate counterfactuals
    counterfactual_info = generate_all_counterfactuals(
        model, 
        test_loader, 
        device, 
        num_classes, 
        test_dataset.class_names, 
        args.output_dir,
        max_samples=args.max_samples,
        skip_optimization=args.skip_optimization
    )
    
    print(f"\nDiCE counterfactual analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF