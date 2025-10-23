import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import warnings
import time
warnings.filterwarnings('ignore')

torch.set_num_threads(os.cpu_count())

# ==================== Configuration ====================

DATA_DIR = '/Users/wynndiaz/Yi He Laboratory/GluttonData'
CSV_FILES = ['1.csv', '2.csv', '3.csv']
BATCH_SIZE = 2048
EPOCHS = 150
LEARNING_RATE = 0.0005
DEVICE = 'mps'
SEED = 42
LOG_INTERVAL = 10

# Architecture
CONTEXT_WINDOW = 3  # Use ±3 residues (total window = 7)
CONV_CHANNELS = [64, 128, 256]
KERNEL_SIZE = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== Data Loading ====================

def load_and_preprocess_data(data_dir, csv_files):
    """Load and clean CSV data with masking for missing values"""
    all_dfs = []

    for csv_file in csv_files:
        data_path = os.path.join(data_dir, csv_file)
        print(f"Loading {csv_file}...")

        names = ['RES1', 'H', 'N', 'HA', 'CA', 'Cp', 'SS', 'PHI', 'PSI']
        df = pd.read_csv(data_path, names=names, header=None)
        print(f"  Raw shape: {df.shape}")
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined: {df.shape}")

    # Drop rows with NaN values first (actual NaN in CSV, not 9999)
    before = len(df)
    df = df.dropna()
    if before - len(df) > 0:
        print(f"Dropped {before - len(df)} rows with NaN values")

    # Create mask columns BEFORE replacing 9999
    for col in ['H', 'N', 'HA', 'CA', 'Cp']:
        mask_col = f'{col}_mask'
        df[mask_col] = (df[col] != 9999.0).astype(np.float32)

    # Replace 9999 with 0 (but keep the rows!)
    for col in ['H', 'N', 'HA', 'CA', 'Cp']:
        df[col] = df[col].replace(9999.0, 0.0)

    # Create mask for valid angles (1 if valid, 0 if 360 or missing)
    df['angles_valid'] = ((df['PHI'].abs() < 360) & (df['PSI'].abs() < 360)).astype(np.float32)
    invalid_count = (df['angles_valid'] == 0).sum()
    print(f"Rows with invalid angles (360°): {invalid_count}")

    # Replace 360 angles with 0 (but keep the rows for context!)
    df['PHI'] = df['PHI'].replace(360.0, 0.0).replace(-360.0, 0.0)
    df['PSI'] = df['PSI'].replace(360.0, 0.0).replace(-360.0, 0.0)

    print(f"Final dataset: {df.shape}")
    return df


# ==================== Dataset ====================

class ContextWindowDataset(Dataset):
    """Dataset with context window"""

    def __init__(self, df, context_window=3):
        self.df = df.reset_index(drop=True)
        self.context_window = context_window
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

        # Only keep residues with:
        # 1. Sufficient context (within dataframe bounds)
        # 2. Valid angles at center (not 360°)
        self.valid_indices = []
        for i in range(context_window, len(df) - context_window):
            # Check if center residue has valid angles
            if df.iloc[i]['angles_valid'] == 1:
                self.valid_indices.append(i)

        total_possible = len(df) - 2 * context_window
        print(f"Valid samples with context (±{context_window}) and valid angles: {len(self.valid_indices)}/{total_possible}")

    def _encode_residue(self, row):
        """Encode single residue: 20 AA + 5 chem shifts + 5 masks = 30 features"""
        aa_encoding = np.zeros(20, dtype=np.float32)
        if row['RES1'] in self.aa_to_idx:
            aa_encoding[self.aa_to_idx[row['RES1']]] = 1.0

        cs = np.array([row['H'], row['N'], row['HA'], row['CA'], row['Cp']],
                     dtype=np.float32)
        masks = np.array([row['H_mask'], row['N_mask'], row['HA_mask'],
                         row['CA_mask'], row['Cp_mask']], dtype=np.float32)

        return np.concatenate([aa_encoding, cs, masks])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]

        # Get context window features
        features = []
        for offset in range(-self.context_window, self.context_window + 1):
            res_idx = center_idx + offset
            features.append(self._encode_residue(self.df.iloc[res_idx]))

        features = np.stack(features)  # Shape: (2*window+1, 30)

        # Get target angles for center residue
        row = self.df.iloc[center_idx]
        phi_deg = row['PHI']
        psi_deg = row['PSI']

        # Safety check - this should never trigger if filtering is correct
        if abs(phi_deg) >= 360 or abs(psi_deg) >= 360:
            raise ValueError(f"Invalid angle in dataset! PHI={phi_deg}, PSI={psi_deg} at index {center_idx}")

        angles = np.array([np.deg2rad(phi_deg), np.deg2rad(psi_deg)],
                         dtype=np.float32)

        # Validate no NaN/Inf in features or angles
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError(f"NaN/Inf in features at index {center_idx}")
        if np.isnan(angles).any() or np.isinf(angles).any():
            raise ValueError(f"NaN/Inf in angles at index {center_idx}: PHI={phi_deg}, PSI={psi_deg}")

        return torch.FloatTensor(features), torch.FloatTensor(angles)


# ==================== CNN Architecture ====================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        # Use GroupNorm instead of BatchNorm - more stable for small/weird batches
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


class ContextCNN(nn.Module):
    """CNN that processes context window to predict angles"""

    def __init__(self, in_feat=30, channels=[64, 128, 256], kernel=3, dropout=0.3):
        super().__init__()

        # Conv layers
        layers = []
        in_ch = in_feat
        for out_ch in channels:
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Attention pooling over sequence
        self.attn = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat)
        x = x.transpose(1, 2)  # (batch, feat, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # Attention pooling
        attn_w = F.softmax(self.attn(x), dim=1)
        x = torch.sum(x * attn_w, dim=1)  # (batch, channels)

        # Predict angles
        angles = self.head(x)
        angles = torch.tanh(angles) * np.pi  # Constrain to [-π, π]

        return angles


# ==================== Loss ====================

class CircularMSELoss(nn.Module):
    """MSE loss for circular data (angles) with numerical stability"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Circular distance using stable formulation
        pred, target: (batch, 2) for phi and psi

        Uses: loss = 2*(1 - cos(target - pred))
        This is equivalent to ||e^(i*target) - e^(i*pred)||^2
        and is numerically stable (no atan2 gradients)
        """
        # Compute difference
        raw_diff = target - pred

        # Circular MSE: 2 * (1 - cos(diff))
        # This is bounded [0, 4] and has stable gradients
        cos_diff = torch.cos(raw_diff)
        circular_mse = 2.0 * (1.0 - cos_diff)

        return torch.mean(circular_mse)


# ==================== Training ====================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total = 0.0
    running = 0.0
    grad_norms = []
    batch_times = []
    start_time = time.time()

    for batch_idx, (feat, angles) in enumerate(loader):
        batch_start = time.time()
        feat = feat.to(device)
        angles = angles.to(device)

        optimizer.zero_grad()

        pred = model(feat)

        loss = criterion(pred, angles)

        # Check for NaN loss before backward
        if torch.isnan(loss):
            print(f"\n!!! NaN loss detected at batch {batch_idx}")
            print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"Pred mean: {pred.mean().item():.4f}, std: {pred.std().item():.4f}")
            print(f"Target range: [{angles.min().item():.4f}, {angles.max().item():.4f}]")
            print(f"Target mean: {angles.mean().item():.4f}, std: {angles.std().item():.4f}")
            print(f"Features range: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
            raise ValueError("NaN loss encountered - stopping training")

        loss.backward()

        # More aggressive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Check for NaN gradients
        if torch.isnan(grad_norm):
            print(f"\n!!! NaN gradients at batch {batch_idx}")
            print(f"Loss was: {loss.item():.4f}")
            # Check which parameters have NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"  NaN in gradients of: {name}")
            raise ValueError("NaN gradients encountered - stopping training")

        optimizer.step()

        total += loss.item()
        running += loss.item()
        grad_norms.append(grad_norm.item())
        batch_times.append(time.time() - batch_start)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg = running / LOG_INTERVAL
            avg_time = np.mean(batch_times[-LOG_INTERVAL:])
            samples_per_sec = BATCH_SIZE / avg_time
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}] Loss: {avg:.4f} | GradNorm: {grad_norm:.4f} | {samples_per_sec:.0f} samples/s')
            running = 0.0

    epoch_time = time.time() - start_time
    return {
        'loss': total / len(loader),
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_std': np.std(grad_norms),
        'grad_norm_max': np.max(grad_norms),
        'epoch_time': epoch_time,
        'samples_per_sec': len(loader.dataset) / epoch_time,
        'avg_batch_time': np.mean(batch_times)
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    start_time = time.time()

    all_pred = []
    all_target = []

    with torch.no_grad():
        for feat, angles in loader:
            feat = feat.to(device)
            angles = angles.to(device)

            pred = model(feat)
            loss = criterion(pred, angles)
            total += loss.item()

            all_pred.append(pred.cpu().numpy())
            all_target.append(angles.cpu().numpy())

    eval_time = time.time() - start_time
    pred = np.vstack(all_pred)
    target = np.vstack(all_target)

    # Calculate MAE
    phi_err = np.abs(np.angle(np.exp(1j * (target[:, 0] - pred[:, 0]))))
    psi_err = np.abs(np.angle(np.exp(1j * (target[:, 1] - pred[:, 1]))))

    phi_mae = np.rad2deg(np.mean(phi_err))
    psi_mae = np.rad2deg(np.mean(psi_err))

    phi_rmse = np.rad2deg(np.sqrt(np.mean(phi_err**2)))
    psi_rmse = np.rad2deg(np.sqrt(np.mean(psi_err**2)))

    # Additional statistics
    phi_median = np.rad2deg(np.median(phi_err))
    psi_median = np.rad2deg(np.median(psi_err))

    phi_p90 = np.rad2deg(np.percentile(phi_err, 90))
    psi_p90 = np.rad2deg(np.percentile(psi_err, 90))

    phi_p95 = np.rad2deg(np.percentile(phi_err, 95))
    psi_p95 = np.rad2deg(np.percentile(psi_err, 95))

    return {
        'loss': total / len(loader),
        'phi_mae': phi_mae,
        'psi_mae': psi_mae,
        'phi_rmse': phi_rmse,
        'psi_rmse': psi_rmse,
        'phi_median': phi_median,
        'psi_median': psi_median,
        'phi_p90': phi_p90,
        'psi_p90': psi_p90,
        'phi_p95': phi_p95,
        'psi_p95': psi_p95,
        'eval_time': eval_time,
        'pred': pred,
        'target': target
    }


def plot_results(results, save_path='results_simple.png'):
    """Plot predictions vs targets"""
    pred = results['pred']
    target = results['target']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Phi
    ax = axes[0]
    ax.scatter(np.rad2deg(target[:, 0]), np.rad2deg(pred[:, 0]), 
               alpha=0.1, s=1)
    ax.plot([-180, 180], [-180, 180], 'r--', lw=2)
    ax.set_xlabel('True Phi (°)')
    ax.set_ylabel('Predicted Phi (°)')
    ax.set_title(f'Phi (MAE={results["phi_mae"]:.1f}°)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    # Psi
    ax = axes[1]
    ax.scatter(np.rad2deg(target[:, 1]), np.rad2deg(pred[:, 1]), 
               alpha=0.1, s=1)
    ax.plot([-180, 180], [-180, 180], 'r--', lw=2)
    ax.set_xlabel('True Psi (°)')
    ax.set_ylabel('Predicted Psi (°)')
    ax.set_title(f'Psi (MAE={results["psi_mae"]:.1f}°)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    # Ramachandran comparison
    ax = axes[2]
    ax.hexbin(np.rad2deg(target[:, 0]), np.rad2deg(target[:, 1]), 
              gridsize=50, cmap='Blues', alpha=0.6, mincnt=1)
    ax.hexbin(np.rad2deg(pred[:, 0]), np.rad2deg(pred[:, 1]), 
              gridsize=50, cmap='Reds', alpha=0.4, mincnt=1)
    ax.set_xlabel('Phi (°)')
    ax.set_ylabel('Psi (°)')
    ax.set_title('Ramachandran (Blue=True, Red=Pred)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


# ==================== Main ====================

def main():
    print("="*80)
    print("GLUTTON CNN - Protein-Aware Context Window")
    print("="*80)
    print(f"Architecture: 1D CNN with attention")
    print(f"Loss: Circular MSE")
    print(f"Context: ±{CONTEXT_WINDOW} residues")
    print(f"Features: 20 AA + 5 chem shifts + 5 masks = 30 per residue")
    print(f"Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("="*80)

    # Load data
    df = load_and_preprocess_data(DATA_DIR, CSV_FILES)

    # Contiguous train-test split to preserve context integrity
    print("\n" + "="*80)
    print("Contiguous Train-Test Split")
    print("="*80)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {len(train_df)} residues (0 to {split_idx-1})")
    print(f"Test: {len(test_df)} residues ({split_idx} to {len(df)-1})")
    print(f"No overlap in context windows between train and test")

    # Create datasets
    print("\n" + "="*80)
    print("Creating Datasets")
    print("="*80)
    train_ds = ContextWindowDataset(train_df, CONTEXT_WINDOW)
    test_ds = ContextWindowDataset(test_df, CONTEXT_WINDOW)

    # AIDEV-NOTE: Optimized DataLoader for IO performance on M4 Max (16 cores, 128GB RAM)
    # Loaders with optimized settings to reduce IO bottleneck
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=14,  # Use more cores (leave 2 for main process)
        pin_memory=True,  # Faster transfer to MPS device
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Load 4 batches per worker ahead of time
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,  # Fewer workers for eval (less memory pressure)
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Model
    model = ContextCNN(in_feat=30, channels=CONV_CHANNELS,
                       kernel=KERNEL_SIZE, dropout=0.3).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    criterion = CircularMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      patience=2, factor=0.5)

    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---')

        train_stats = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        res = evaluate(model, test_loader, criterion, DEVICE)

        train_losses.append(train_stats['loss'])
        val_losses.append(res['loss'])

        scheduler.step(res['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        if res['loss'] < best:
            best = res['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': res['loss']
            }, 'glutton_simple_best.pth')
            print('  ★ Best model saved')

        # AIDEV-NOTE: Comprehensive reporting with loss, angle metrics, gradient stats, timing
        # Comprehensive epoch-end reporting
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{EPOCHS} SUMMARY")
        print(f"{'='*80}")

        # Loss metrics
        print(f"\nLoss Metrics:")
        print(f"  Train Loss: {train_stats['loss']:.4f} | Val Loss: {res['loss']:.4f} | Best: {best:.4f}")

        # Angle prediction metrics
        print(f"\nAngle Prediction (degrees):")
        print(f"  Phi   - MAE: {res['phi_mae']:6.2f}° | RMSE: {res['phi_rmse']:6.2f}° | Median: {res['phi_median']:6.2f}° | 90th: {res['phi_p90']:6.2f}° | 95th: {res['phi_p95']:6.2f}°")
        print(f"  Psi   - MAE: {res['psi_mae']:6.2f}° | RMSE: {res['psi_rmse']:6.2f}° | Median: {res['psi_median']:6.2f}° | 90th: {res['psi_p90']:6.2f}° | 95th: {res['psi_p95']:6.2f}°")
        print(f"  Average MAE: {(res['phi_mae'] + res['psi_mae'])/2:.2f}°")

        # Training dynamics
        print(f"\nTraining Dynamics:")
        print(f"  Gradient Norm - Mean: {train_stats['grad_norm_mean']:.4f} | Std: {train_stats['grad_norm_std']:.4f} | Max: {train_stats['grad_norm_max']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Performance metrics
        print(f"\nPerformance:")
        print(f"  Train - {train_stats['epoch_time']:.1f}s | {train_stats['samples_per_sec']:.0f} samples/s | {train_stats['avg_batch_time']*1000:.1f}ms/batch")
        print(f"  Eval  - {res['eval_time']:.1f}s")
        print(f"  Total - {train_stats['epoch_time'] + res['eval_time']:.1f}s")

        print(f"{'='*80}\n")
    
    # Final eval
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    checkpoint = torch.load('glutton_simple_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    res = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Test Loss: {res['loss']:.4f}")
    print(f"Phi MAE: {res['phi_mae']:.2f}°")
    print(f"Psi MAE: {res['psi_mae']:.2f}°")
    print(f"Avg MAE: {(res['phi_mae'] + res['psi_mae'])/2:.2f}°")
    
    # Plot
    plot_results(res)
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved: training_curves.png")
    plt.close()
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    
    return model, res


if __name__ == "__main__":
    model, results = main()