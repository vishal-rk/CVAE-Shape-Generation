cat > cvae_confusion_matrix.py << 'EOF'
import os
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

def generate_confusion_matrix(model, test_loader, device, num_classes, class_names, output_dir, max_samples=200):
    """
    Generate a confusion matrix based on reconstruction quality.
    For each image, find which class condition produces the best reconstruction.
    """
    print("Generating confusion matrix...")
    confusion_dir = os.path.join(output_dir, 'confusion_matrix')
    os.makedirs(confusion_dir, exist_ok=True)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track IoU values for each class pair
    class_pair_ious = {(orig, pred): [] for orig in range(num_classes) for pred in range(num_classes)}
    
    # Track processed samples per class
    processed_per_class = {i: 0 for i in range(num_classes)}
    
    model.eval()
    with torch.no_grad():
        for data, labels, filenames in tqdm(test_loader, desc="Computing confusion matrix"):
            data = data.to(device)
            labels = labels.to(device)
            
            # Process each image
            for i, (img, label, filename) in enumerate(zip(data, labels, filenames)):
                true_class = label.item()
                
                # Skip if we've processed enough from this class
                if processed_per_class[true_class] >= max_samples // num_classes:
                    continue
                
                processed_per_class[true_class] += 1
                img_expanded = img.unsqueeze(0)  # Add batch dimension
                
                best_iou = -1
                predicted_class = -1
                iou_scores = []
                reconstructions = []
                
                # Test reconstruction with each class condition
                for class_idx in range(num_classes):
                    # Create class condition
                    class_condition = torch.zeros(1, num_classes, device=device)
                    class_condition[0, class_idx] = 1.0
                    
                    # Reconstruct the image with this class condition
                    recon, _, _ = model(img_expanded, class_condition)
                    
                    # Calculate IoU
                    iou = calculate_iou(recon.cpu(), img_expanded.cpu())
                    iou_scores.append(iou)
                    reconstructions.append(recon.cpu())
                    
                    # Record IoU for this class pair
                    class_pair_ious[(true_class, class_idx)].append(iou)
                    
                    # Update best prediction
                    if iou > best_iou:
                        best_iou = iou
                        predicted_class = class_idx
                
                # Update confusion matrix
                confusion_matrix[true_class, predicted_class] += 1
                
                # Save misclassifications for analysis
                if true_class != predicted_class:
                    # Create visualization of original vs. reconstructions
                    fig, axes = plt.subplots(1, num_classes + 1, figsize=(3*(num_classes + 1), 3))
                    
                    # Plot original image
                    axes[0].imshow(img.cpu().squeeze().numpy(), cmap='gray')
                    axes[0].set_title(f"Original\n({class_names[true_class]})")
                    axes[0].axis('off')
                    
                    # Plot all reconstructions
                    for j, (recon, iou_val) in enumerate(zip(reconstructions, iou_scores)):
                        axes[j+1].imshow(recon.squeeze().numpy(), cmap='gray')
                        axes[j+1].set_title(f"As {class_names[j]}\nIoU: {iou_val:.4f}")
                        axes[j+1].axis('off')
                        
                        # Highlight the predicted class
                        if j == predicted_class:
                            axes[j+1].set_title(f"As {class_names[j]}\nIoU: {iou_val:.4f}", fontweight='bold', color='red')
                            for spine in axes[j+1].spines.values():
                                spine.set_edgecolor('red')
                                spine.set_linewidth(2)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(confusion_dir, f"misclassified_{class_names[true_class]}_as_{class_names[predicted_class]}_{filename}.png"),
                                dpi=150, bbox_inches='tight')
                    plt.close()
    
    # Calculate and save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class (Best Reconstruction)')
    plt.ylabel('True Class')
    plt.title('Reconstruction-Based Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(confusion_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Calculate and save normalized confusion matrix
    row_sums = confusion_matrix.sum(axis=1)
    norm_confusion = np.zeros_like(confusion_matrix, dtype=float)
    for i in range(num_classes):
        if row_sums[i] > 0:
            norm_confusion[i, :] = confusion_matrix[i, :] / row_sums[i]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_confusion, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class (Best Reconstruction)')
    plt.ylabel('True Class')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(confusion_dir, 'normalized_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Create average IoU heatmap between classes
    avg_iou_matrix = np.zeros((num_classes, num_classes))
    for (orig, pred), ious in class_pair_ious.items():
        if ious:  # Check if list is not empty
            avg_iou_matrix[orig, pred] = np.mean(ious)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_iou_matrix, annot=True, fmt='.4f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Reconstruction Condition Class')
    plt.ylabel('Original Image Class')
    plt.title('Average IoU Between Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(confusion_dir, 'average_iou_matrix.png'), dpi=300)
    plt.close()
    
    # Save summary statistics
    with open(os.path.join(confusion_dir, 'confusion_matrix_analysis.txt'), 'w') as f:
        f.write("Confusion Matrix Analysis\n")
        f.write("=======================\n\n")
        
        # Calculate accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Performance:\n")
        for i in range(num_classes):
            precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum() if confusion_matrix[:, i].sum() > 0 else 0
            recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write(f"{class_names[i]}:\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  Samples: {processed_per_class[i]}\n\n")
    
    print(f"Confusion matrix analysis saved to {confusion_dir}")
    return confusion_matrix

def main():
    parser = argparse.ArgumentParser(description='CVAE Confusion Matrix Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--data-dir', type=str, default='test_dataset',
                        help='Root directory of the test dataset')
    parser.add_argument('--output-dir', type=str, default='cvae_confusion_matrix',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--latent-size', type=int, default=100,
                        help='Size of the latent space')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate (train, val, test)')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Generate confusion matrix
    confusion_matrix = generate_confusion_matrix(
        model, 
        test_loader, 
        device, 
        num_classes, 
        test_dataset.class_names, 
        args.output_dir,
        max_samples=args.max_samples
    )
    
    print(f"\nConfusion matrix analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF