import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv
import json

# Image size
IMAGE_SIZE = 512

# Define output directories
BASE_DIR = 'cvae_demo_xai'
RESULTS_DIR = f'{BASE_DIR}/cvae_512_results'
CHECKPOINT_DIR = f'{BASE_DIR}/cvae_512_checkpoints'
LOSS_DIR = f'{BASE_DIR}/cvae_512_losses'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(f'{LOSS_DIR}/batch_losses', exist_ok=True)
os.makedirs(f'{LOSS_DIR}/epoch_losses', exist_ok=True)
os.makedirs(f'{LOSS_DIR}/components', exist_ok=True)

# Dataset class adapted for server's folder structure
class ServerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_names, split='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        exts = {'.bmp', '.png', '.jpg', '.jpeg', '.gif'}
        
        # Path format differs from original code
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
                        print(f"Warning: Directory not found, skipping: {split_dir}")
                        continue
                    
                    # Get all image files in that directory
                    for fname in sorted(os.listdir(split_dir)):
                        if os.path.splitext(fname.lower())[1] in exts:
                            path = os.path.join(split_dir, fname)
                            self.image_paths.append(path)
                            self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('L')  # force 1-channel
            if self.transform is not None:
                img = self.transform(img)
            # Ensure pixel values are in valid range [0,1]
            img = torch.clamp(img, 0.0, 1.0)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return img, label

# Convolutional CVAE for 512x512 images
class CVAE(nn.Module):
    def __init__(self, latent_size, class_size):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.class_size = class_size
        
        # Store image size for XAI functions
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
        # Ensure reconstructed values are in valid range [0,1]
        recon_x = torch.clamp(recon_x, 0.0, 1.0)
        return recon_x, mu, logvar

# Helper and Loss functions
def one_hot(labels, class_size, device):
    targets = torch.zeros(labels.size(0), class_size, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return targets

def detailed_loss_function(recon_x, x, mu, logvar):
    """Calculate BCE, KL, and MSE losses separately"""
    # Ensure inputs are in valid range [0,1]
    recon_x_safe = torch.clamp(recon_x.view(-1, IMAGE_SIZE * IMAGE_SIZE), 0.0, 1.0)
    x_safe = torch.clamp(x.view(-1, IMAGE_SIZE * IMAGE_SIZE), 0.0, 1.0)
    
    try:
        # Binary Cross Entropy 
        BCE = F.binary_cross_entropy(recon_x_safe, x_safe, reduction='sum')
        
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Mean Squared Error
        MSE = F.mse_loss(recon_x_safe, x_safe, reduction='sum')
        
        # Return all components and total loss
        return BCE + KLD, BCE.item(), KLD.item(), MSE.item()
    except RuntimeError as e:
        print(f"Error in loss calculation: {e}")
        print(f"recon_x range: {recon_x_safe.min().item()} to {recon_x_safe.max().item()}")
        print(f"x range: {x_safe.min().item()} to {x_safe.max().item()}")
        # Return default values
        default_loss = torch.tensor(1e6, device=recon_x.device, dtype=recon_x.dtype)
        return default_loss, 1e6, 1e6, 1e6

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU for binary masks"""
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

def calculate_batch_iou(predictions, targets, threshold=0.5):
    """Calculate IoU for a batch of images"""
    batch_ious = []
    
    for i in range(predictions.size(0)):
        pred = predictions[i].view(IMAGE_SIZE, IMAGE_SIZE)
        target = targets[i].view(IMAGE_SIZE, IMAGE_SIZE)
        iou = calculate_iou(pred, target, threshold)
        batch_ious.append(iou)
    
    return batch_ious, torch.tensor(batch_ious).mean().item()

# Save functions for different levels of losses
def save_batch_losses(epoch, batch_losses, loss_type, split='train'):
    """Save per-batch losses to files"""
    # Save as text file
    batch_file = os.path.join(LOSS_DIR, 'batch_losses', f'{split}_{loss_type}_batch_losses_epoch_{epoch}.txt')
    with open(batch_file, 'w') as f:
        f.write(f"{split.capitalize()} {loss_type.upper()} Batch Losses - Epoch {epoch}\n")
        f.write("=" * 40 + "\n")
        f.write("Batch_Index,Loss_Value\n")
        for i, loss in enumerate(batch_losses):
            f.write(f"{i+1},{loss:.6f}\n")
        
        # Add statistics
        if batch_losses:
            f.write(f"\nStatistics:\n")
            f.write(f"Average: {sum(batch_losses)/len(batch_losses):.6f}\n")
            f.write(f"Min: {min(batch_losses):.6f}\n")
            f.write(f"Max: {max(batch_losses):.6f}\n")
            f.write(f"Std Dev: {np.std(batch_losses):.6f}\n")
    
    # Save as CSV for easier analysis
    csv_file = os.path.join(LOSS_DIR, 'batch_losses', f'{split}_{loss_type}_batch_losses_epoch_{epoch}.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_index', 'loss_value'])
        for i, loss in enumerate(batch_losses):
            writer.writerow([i+1, loss])

def save_epoch_losses(epoch, train_loss, test_loss, val_loss, components):
    """Save epoch-level losses with components"""
    epoch_file = os.path.join(LOSS_DIR, 'epoch_losses', f'epoch_{epoch}_losses.txt')
    with open(epoch_file, 'w') as f:
        f.write(f"Epoch {epoch} Loss Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Train Loss: {train_loss:.6f}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Validation Loss: {val_loss:.6f}\n")
        f.write(f"Total Loss: {train_loss + test_loss + val_loss:.6f}\n\n")
        
        f.write("Loss Components:\n")
        f.write("-" * 30 + "\n")
        for split in ['train', 'test', 'val']:
            f.write(f"{split.capitalize()} BCE: {components[split]['bce']:.6f}\n")
            f.write(f"{split.capitalize()} KL: {components[split]['kl']:.6f}\n")
            f.write(f"{split.capitalize()} MSE: {components[split]['mse']:.6f}\n")
            if 'iou' in components[split]:
                f.write(f"{split.capitalize()} IoU: {components[split]['iou']:.6f}\n")
            f.write("-" * 30 + "\n")

def save_complete_loss_history(history):
    """Save complete loss history to a single file"""
    history_file = os.path.join(LOSS_DIR, 'complete_loss_history.txt')
    with open(history_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTest_Loss\tVal_Loss\tTrain_BCE\tTrain_KL\tTrain_MSE\tTest_BCE\tTest_KL\tTest_MSE\tVal_BCE\tVal_KL\tVal_MSE\tVal_IoU\n")
        for i in range(len(history['epochs'])):
            epoch = history['epochs'][i]
            f.write(f"{epoch}\t{history['train']['total'][i]:.6f}\t{history['test']['total'][i]:.6f}\t{history['val']['total'][i]:.6f}\t")
            f.write(f"{history['train']['bce'][i]:.6f}\t{history['train']['kl'][i]:.6f}\t{history['train']['mse'][i]:.6f}\t")
            f.write(f"{history['test']['bce'][i]:.6f}\t{history['test']['kl'][i]:.6f}\t{history['test']['mse'][i]:.6f}\t")
            f.write(f"{history['val']['bce'][i]:.6f}\t{history['val']['kl'][i]:.6f}\t{history['val']['mse'][i]:.6f}\t")
            if 'iou' in history['val'] and i < len(history['val']['iou']):
                f.write(f"{history['val']['iou'][i]:.6f}\n")
            else:
                f.write("N/A\n")

# Training function with detailed loss tracking
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    bce_loss_sum = 0
    kl_loss_sum = 0
    mse_loss_sum = 0
    
    # Track individual batch losses
    total_batch_losses = []
    bce_batch_losses = []
    kl_batch_losses = []
    mse_batch_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Train Epoch: {epoch}")
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        labels_onehot = one_hot(labels, model.class_size, device)
        
        optimizer.zero_grad()
        
        try:
            recon_batch, mu, logvar = model(data, labels_onehot)
            loss, bce, kl, mse = detailed_loss_function(recon_batch, data, mu, logvar)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()} at batch {batch_idx}, skipping backward pass")
                continue
                
            loss.backward()
            
            # Accumulate losses
            train_loss += loss.item()
            bce_loss_sum += bce
            kl_loss_sum += kl
            mse_loss_sum += mse
            
            # Store per-batch losses normalized by batch size
            batch_size = len(data)
            total_batch_losses.append(loss.item() / batch_size)
            bce_batch_losses.append(bce / batch_size)
            kl_batch_losses.append(kl / batch_size)
            mse_batch_losses.append(mse / batch_size)
            
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() / batch_size,
                'bce': bce / batch_size,
                'kl': kl / batch_size,
                'mse': mse / batch_size
            })
            
            if batch_idx % 20 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / batch_size:.6f}, '
                      f'BCE: {bce / batch_size:.6f}, KL: {kl / batch_size:.6f}, MSE: {mse / batch_size:.6f}')
        
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue
    
    # Save per-batch losses to files
    save_batch_losses(epoch, total_batch_losses, 'total', 'train')
    save_batch_losses(epoch, bce_batch_losses, 'bce', 'train')
    save_batch_losses(epoch, kl_batch_losses, 'kl', 'train')
    save_batch_losses(epoch, mse_batch_losses, 'mse', 'train')
    
    # Calculate average losses
    n_samples = len(train_loader.dataset)
    avg_loss = train_loss / n_samples
    avg_bce = bce_loss_sum / n_samples
    avg_kl = kl_loss_sum / n_samples
    avg_mse = mse_loss_sum / n_samples
    
    print(f'====> Epoch: {epoch} Average training losses:')
    print(f'      Total: {avg_loss:.6f}, BCE: {avg_bce:.6f}, KL: {avg_kl:.6f}, MSE: {avg_mse:.6f}')
    
    return avg_loss, avg_bce, avg_kl, avg_mse

# Test function with detailed loss tracking
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    bce_loss_sum = 0
    kl_loss_sum = 0
    mse_loss_sum = 0
    
    # Track individual batch losses
    total_batch_losses = []
    bce_batch_losses = []
    kl_batch_losses = []
    mse_batch_losses = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc=f"Test Epoch: {epoch}")):
            data, labels = data.to(device), labels.to(device)
            labels_onehot = one_hot(labels, model.class_size, device)
            
            try:
                recon_batch, mu, logvar = model(data, labels_onehot)
                loss, bce, kl, mse = detailed_loss_function(recon_batch, data, mu, logvar)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    test_loss += loss.item()
                    bce_loss_sum += bce
                    kl_loss_sum += kl
                    mse_loss_sum += mse
                    
                    # Store per-batch losses normalized by batch size
                    batch_size = len(data)
                    total_batch_losses.append(loss.item() / batch_size)
                    bce_batch_losses.append(bce / batch_size)
                    kl_batch_losses.append(kl / batch_size)
                    mse_batch_losses.append(mse / batch_size)
                
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([
                        data[:n], 
                        recon_batch.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)[:n]
                    ])
                    comparison = torch.clamp(comparison, 0.0, 1.0)
                    save_image(comparison.cpu(), 
                              os.path.join(RESULTS_DIR, f'reconstruction_epoch_{epoch}.png'), 
                              nrow=n)
            except RuntimeError as e:
                print(f"Error in test batch {i}: {e}")
                continue
    
    # Save per-batch losses to files
    save_batch_losses(epoch, total_batch_losses, 'total', 'test')
    save_batch_losses(epoch, bce_batch_losses, 'bce', 'test')
    save_batch_losses(epoch, kl_batch_losses, 'kl', 'test')
    save_batch_losses(epoch, mse_batch_losses, 'mse', 'test')
    
    # Calculate average losses
    n_samples = len(test_loader.dataset)
    avg_loss = test_loss / n_samples
    avg_bce = bce_loss_sum / n_samples
    avg_kl = kl_loss_sum / n_samples
    avg_mse = mse_loss_sum / n_samples
    
    print(f'====> Test set losses:')
    print(f'      Total: {avg_loss:.6f}, BCE: {avg_bce:.6f}, KL: {avg_kl:.6f}, MSE: {avg_mse:.6f}')
    
    return avg_loss, avg_bce, avg_kl, avg_mse

# Validation function with IoU calculation
def validate(model, device, val_loader, epoch):
    model.eval()
    val_loss = 0
    bce_loss_sum = 0
    kl_loss_sum = 0
    mse_loss_sum = 0
    total_iou = 0.0
    num_batches = 0
    
    # Track individual batch losses and IoU values
    total_batch_losses = []
    bce_batch_losses = []
    kl_batch_losses = []
    mse_batch_losses = []
    iou_batch_values = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch: {epoch}")):
            try:
                data, labels = data.to(device), labels.to(device)
                labels_onehot = one_hot(labels, model.class_size, device)
                recon, mu, logvar = model(data, labels_onehot)
                
                # Calculate losses
                loss, bce, kl, mse = detailed_loss_function(recon, data, mu, logvar)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    bce_loss_sum += bce
                    kl_loss_sum += kl
                    mse_loss_sum += mse
                    
                    # Calculate IoU for this batch
                    batch_ious, batch_iou_avg = calculate_batch_iou(recon.cpu(), data.cpu())
                    total_iou += batch_iou_avg
                    num_batches += 1
                    
                    # Store per-batch losses normalized by batch size
                    batch_size = len(data)
                    total_batch_losses.append(loss.item() / batch_size)
                    bce_batch_losses.append(bce / batch_size)
                    kl_batch_losses.append(kl / batch_size)
                    mse_batch_losses.append(mse / batch_size)
                    iou_batch_values.extend(batch_ious)  # Store individual image IoUs
            
            except RuntimeError as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Save per-batch losses to files
    save_batch_losses(epoch, total_batch_losses, 'total', 'val')
    save_batch_losses(epoch, bce_batch_losses, 'bce', 'val')
    save_batch_losses(epoch, kl_batch_losses, 'kl', 'val')
    save_batch_losses(epoch, mse_batch_losses, 'mse', 'val')
    save_batch_losses(epoch, iou_batch_values, 'iou', 'val')
    
    # Calculate average metrics
    n_samples = len(val_loader.dataset)
    avg_loss = val_loss / n_samples
    avg_bce = bce_loss_sum / n_samples
    avg_kl = kl_loss_sum / n_samples
    avg_mse = mse_loss_sum / n_samples
    avg_iou = total_iou / max(1, num_batches)  # Avoid division by zero
    
    print(f'====> Validation set metrics:')
    print(f'      Total: {avg_loss:.6f}, BCE: {avg_bce:.6f}, KL: {avg_kl:.6f}, MSE: {avg_mse:.6f}')
    print(f'      IoU: {avg_iou:.6f}')
    
    return avg_loss, avg_bce, avg_kl, avg_mse, avg_iou

# Entry point
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reduced workers and no pin_memory to avoid memory issues while keeping 512x512 resolution
    kwargs = {'num_workers': 1, 'pin_memory': False}

    # Hyperparameters - Reduced batch size for stability
    batch_size = 8  # Reduced from 16 to 8 for better stability
    latent_size = 100
    epochs = 30
    lr = 1e-4

    # Loss history dictionary
    loss_history = {
        'epochs': [],
        'train': {'total': [], 'bce': [], 'kl': [], 'mse': []},
        'test': {'total': [], 'bce': [], 'kl': [], 'mse': []},
        'val': {'total': [], 'bce': [], 'kl': [], 'mse': [], 'iou': []}
    }

    # Try to load loss history if it exists
    history_file = os.path.join(LOSS_DIR, 'loss_history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                loss_history = json.load(f)
            print(f"Loaded existing loss history from {history_file}")
        except Exception as e:
            print(f"Error loading loss history: {e}")
            print("Starting with new loss history")

    # Try to load the last checkpoint if it exists to resume training
    start_epoch = 1
    checkpoint_path = None
    for i in range(30, 0, -1):
        check_path = os.path.join(CHECKPOINT_DIR, f'cvae_model_epoch_{i}.pth')
        if os.path.exists(check_path):
            checkpoint_path = check_path
            start_epoch = i + 1
            print(f"Found checkpoint at epoch {i}, resuming from epoch {start_epoch}")
            break

    # DEFINE DATASET HIERARCHY - for server path
    root_data_dir = "/pitsec_sose2025_team3_2/data"
    class_names = [
        'sized_squares_filled',
        'sized_squares_unfilled',
        'sized_rectangles_filled',
        'sized_rectangles_unfilled'
    ]
    num_classes = len(class_names)

    # Transforms: resize to 512×512, convert to Tensor
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Create Datasets & DataLoaders using the server dataset class
    train_dataset = ServerDataset(root_dir=root_data_dir, class_names=class_names, split='train',
                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = ServerDataset(root_dir=root_data_dir, class_names=class_names, split='test',
                                transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    val_dataset = ServerDataset(root_dir=root_data_dir, class_names=class_names, split='val', 
                               transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"Found {len(train_dataset)} training images and {len(test_dataset)} test images.")
    print(f"Found {len(val_dataset)} validation images.")

    # Check if we have any data
    if len(train_dataset) == 0:
        raise ValueError("No training data found. Please check the data directory structure.")

    # Instantiate model & optimizer
    model = CVAE(latent_size=latent_size, class_size=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load checkpoint if available
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            print(f"Loaded model state from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")

    # Training / Testing Loop - NO XAI HERE
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{epochs}")
        print(f"Results: {RESULTS_DIR} | Losses: {LOSS_DIR}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_bce, train_kl, train_mse = train(model, device, train_loader, optimizer, epoch)
        
        # Testing
        test_loss, test_bce, test_kl, test_mse = test(model, device, test_loader, epoch)
        
        # Validation with IoU
        val_loss, val_bce, val_kl, val_mse, val_iou = validate(model, device, val_loader, epoch)

        # Update loss history
        if epoch not in loss_history['epochs']:
            loss_history['epochs'].append(epoch)
        
        # Find the correct index for this epoch
        idx = loss_history['epochs'].index(epoch)
        
        # Extend the lists if needed
        while len(loss_history['train']['total']) <= idx:
            loss_history['train']['total'].append(0)
            loss_history['train']['bce'].append(0)
            loss_history['train']['kl'].append(0)
            loss_history['train']['mse'].append(0)
            loss_history['test']['total'].append(0)
            loss_history['test']['bce'].append(0)
            loss_history['test']['kl'].append(0)
            loss_history['test']['mse'].append(0)
            loss_history['val']['total'].append(0)
            loss_history['val']['bce'].append(0)
            loss_history['val']['kl'].append(0)
            loss_history['val']['mse'].append(0)
            loss_history['val']['iou'].append(0)
        
        # Store the current epoch's values
        loss_history['train']['total'][idx] = train_loss
        loss_history['train']['bce'][idx] = train_bce
        loss_history['train']['kl'][idx] = train_kl
        loss_history['train']['mse'][idx] = train_mse
        
        loss_history['test']['total'][idx] = test_loss
        loss_history['test']['bce'][idx] = test_bce
        loss_history['test']['kl'][idx] = test_kl
        loss_history['test']['mse'][idx] = test_mse
        
        loss_history['val']['total'][idx] = val_loss
        loss_history['val']['bce'][idx] = val_bce
        loss_history['val']['kl'][idx] = val_kl
        loss_history['val']['mse'][idx] = val_mse
        loss_history['val']['iou'][idx] = val_iou
        
        # Save epoch losses
        loss_components = {
            'train': {'bce': train_bce, 'kl': train_kl, 'mse': train_mse},
            'test': {'bce': test_bce, 'kl': test_kl, 'mse': test_mse},
            'val': {'bce': val_bce, 'kl': val_kl, 'mse': val_mse, 'iou': val_iou}
        }
        save_epoch_losses(epoch, train_loss, test_loss, val_loss, loss_components)
        
        # Save complete history
        save_complete_loss_history(loss_history)
        
        # Save loss history as JSON for easy loading
        with open(history_file, 'w') as f:
            json.dump(loss_history, f, indent=4)

        # Print summary
        print(f"\nLoss Summary for Epoch {epoch}:")
        print(f"  Train - Total: {train_loss:.6f}, BCE: {train_bce:.6f}, KL: {train_kl:.6f}, MSE: {train_mse:.6f}")
        print(f"  Test  - Total: {test_loss:.6f}, BCE: {test_bce:.6f}, KL: {test_kl:.6f}, MSE: {test_mse:.6f}")
        print(f"  Val   - Total: {val_loss:.6f}, BCE: {val_bce:.6f}, KL: {val_kl:.6f}, MSE: {val_mse:.6f}")
        print(f"  Val IoU: {val_iou:.6f}")
        print(f"  Losses saved to: {LOSS_DIR}")

        # SAVE THE MODEL CHECKPOINT
        model_path = os.path.join(CHECKPOINT_DIR, f'cvae_model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'====> Saved model checkpoint to {model_path}')

        # SAMPLE AND SAVE ONE IMAGE PER CLASS
        with torch.no_grad():
            try:
                # Create a one-hot vector for each class
                c_sample = torch.eye(num_classes, device=device)
                # Use the same random noise for all classes to see the conditional effect
                z_sample = torch.randn(1, latent_size, device=device).repeat(num_classes, 1)

                generated = model.decode(z_sample, c_sample).cpu()
                # Ensure generated images are in valid range
                generated = torch.clamp(generated, 0.0, 1.0)

                save_image(
                    generated.view(num_classes, 1, IMAGE_SIZE, IMAGE_SIZE),
                    os.path.join(RESULTS_DIR, f'sample_epoch_{epoch:02d}.png'),
                    nrow=num_classes  # Arrange images in a row
                )
                print(f'====> Saved generated samples for epoch {epoch}')
            except RuntimeError as e:
                print(f"Error generating samples: {e}")

    print("\nTraining complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Losses saved to: {LOSS_DIR}")
    print(f"Model checkpoints saved to: {CHECKPOINT_DIR}")
    print("\nNow you can run the XAI script (cvae_xai.py) to generate explanations for each epoch.")
