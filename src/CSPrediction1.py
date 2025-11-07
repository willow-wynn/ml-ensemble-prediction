import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Enable MPS fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set number of threads to use all CPU cores
torch.set_num_threads(os.cpu_count())

# Configuration - NOW USING TRAINTEST.CSV (90% of data, eval.csv held out)
DATA_PATH = '/Users/wynndiaz/he_lab/data/traintest.csv'
BATCH_SIZE = 128
MAX_SEQ_LEN = 256
EPOCHS = 250
LEARNING_RATE = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MASK_PROB = 0.0

# Model hyperparameters - BIGGER MODEL
CONTEXT_WINDOW = 9
CONV_CHANNELS = [128, 256, 512, 768]  # Added 768 layer!
KERNEL_SIZE = 2
DROPOUT = 0.15

# Wandb config
WANDB_PROJECT = "chemical-shift-prediction"
WANDB_ENTITY = None  # Set to your wandb username if needed
USE_WANDB = True  # Set to False to disable wandb logging

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_split_data(csv_path, test_size=0.15, random_state=42):
    """Load data and split by protein ID"""
    print("Loading data...")
    
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows")
    
    if len(df.columns) != 11:
        raise ValueError(f"Expected 11 columns but got {len(df.columns)}")
    
    df.columns = ['protein_id', 'position', 'residue', 'CS1', 'CS2', 
                  'CS3', 'CS4', 'ss_type', 'mystery_feature', 'phi', 'psi']
    
    df['protein_id'] = pd.to_numeric(df['protein_id'], errors='coerce').astype('Int64')
    df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['protein_id', 'position'])
    
    for col in ['CS1', 'CS2', 'CS3', 'CS4', 'mystery_feature', 'phi', 'psi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    before = len(df)
    df = df.dropna()
    if before > len(df):
        print(f"Dropped {before - len(df)} rows with unparseable data")
    
    print(f"Total rows: {len(df):,}")
    print(f"Total proteins: {df['protein_id'].nunique():,}")
    
    protein_ids = df['protein_id'].unique()
    train_proteins, test_proteins = train_test_split(
        protein_ids, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df['protein_id'].isin(train_proteins)].copy()
    test_df = df[df['protein_id'].isin(test_proteins)].copy()
    
    print(f"\nTrain: {len(train_proteins):,} proteins, {len(train_df):,} residues")
    print(f"Test:  {len(test_proteins):,} proteins, {len(test_df):,} residues")
    
    return train_df, test_df

def compute_stats(df):
    """Compute normalization statistics excluding 9999 values"""
    stats = {}
    cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
    
    print("\nChemical shift statistics (from training data):")
    for col in cs_cols:
        valid = df[df[col] != 9999.0][col]
        stats[col] = {'mean': float(valid.mean()), 'std': float(valid.std())}
        missing_pct = (df[col] == 9999.0).sum() / len(df) * 100
        print(f"{col:15s}: mean={stats[col]['mean']:8.2f}, std={stats[col]['std']:7.2f}, missing={missing_pct:.1f}%")
    
    stats['mystery_feature'] = {
        'mean': float(df['mystery_feature'].mean()),
        'std': float(df['mystery_feature'].std())
    }
    print(f"{'mystery_feature':15s}: mean={stats['mystery_feature']['mean']:8.2f}, std={stats['mystery_feature']['std']:7.2f}")
    
    for col in ['phi', 'psi']:
        valid = df[df[col] != 9999.0][col]
        stats[col] = {'mean': float(valid.mean()), 'std': float(valid.std())}
        missing_pct = (df[col] == 9999.0).sum() / len(df) * 100
        print(f"{col:15s}: mean={stats[col]['mean']:8.2f}, std={stats[col]['std']:7.2f}, missing={missing_pct:.1f}%")
    
    return stats

# ============================================================================
# Dataset with Edge Masking Support
# ============================================================================

class ProteinDataset(Dataset):
    def __init__(self, df, stats, context_window=4, mask_prob=0.2, training=True, 
                 verbose=True, residue_encoder=None, ss_encoder=None, seed=None,
                 guarantee_mask=True):
        self.df = df
        self.stats = stats
        self.context_window = context_window
        self.mask_prob = mask_prob
        self.training = training
        self.seed = seed
        self.guarantee_mask = guarantee_mask
        self.cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
        
        # Use provided encoders or create new ones
        if residue_encoder is not None:
            self.residue_encoder = residue_encoder
        else:
            self.residue_encoder = LabelEncoder()
            all_residues = df['residue'].unique()
            self.residue_encoder.fit(all_residues)
        
        if ss_encoder is not None:
            self.ss_encoder = ss_encoder
        else:
            self.ss_encoder = LabelEncoder()
            all_ss = df['ss_type'].unique()
            self.ss_encoder.fit(all_ss)
        
        self.n_residue_types = len(self.residue_encoder.classes_)
        self.n_ss_types = len(self.ss_encoder.classes_)
        
        if verbose:
            print(f"\nResidue types: {self.n_residue_types} ({list(self.residue_encoder.classes_)[:5]}...)")
            print(f"SS types: {self.n_ss_types} ({list(self.ss_encoder.classes_)})")
        
        # Group by protein
        self.proteins = list(df.groupby('protein_id'))
        
        # Build valid indices - NOW INCLUDING EDGE RESIDUES!
        # We'll mask missing context instead of excluding edges
        self.samples = []
        for protein_id, prot_df in self.proteins:
            prot_df = prot_df.sort_values('position').reset_index(drop=True)
            # Include ALL residues, even at edges
            for i in range(len(prot_df)):
                self.samples.append((protein_id, i))
        
        if verbose:
            print(f"Valid samples (including edges): {len(self.samples):,}")
        
    def normalize(self, value, col):
        if value == 9999.0 or np.isnan(value):
            return 0.0
        return (value - self.stats[col]['mean']) / (self.stats[col]['std'] + 1e-8)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        protein_id, center_idx = self.samples[idx]
        prot_df = self.df[self.df['protein_id'] == protein_id].sort_values('position').reset_index(drop=True)
        
        # Extract context window - handle edges by padding
        start_idx = center_idx - self.context_window
        end_idx = center_idx + self.context_window + 1
        
        # Calculate padding needed
        pad_left = max(0, -start_idx)  # How many positions before start
        pad_right = max(0, end_idx - len(prot_df))  # How many positions after end
        
        # Adjust indices to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(prot_df), end_idx)
        
        window_df = prot_df.iloc[start_idx:end_idx]
        actual_seq_len = len(window_df)
        expected_seq_len = 2 * self.context_window + 1
        
        # Create tensors for full expected length (with padding)
        residues_onehot = torch.zeros(expected_seq_len, self.n_residue_types, dtype=torch.float32)
        ss_onehot = torch.zeros(expected_seq_len, self.n_ss_types, dtype=torch.float32)
        cs_input = torch.zeros(expected_seq_len, 4, dtype=torch.float32)
        cs_target = torch.zeros(expected_seq_len, 4, dtype=torch.float32)
        cs_mask = torch.zeros(expected_seq_len, 4, dtype=torch.bool)
        mystery = torch.zeros(expected_seq_len, dtype=torch.float32)
        phi = torch.zeros(expected_seq_len, dtype=torch.float32)
        psi = torch.zeros(expected_seq_len, dtype=torch.float32)
        
        # Fill in actual data (offset by pad_left)
        for i, res in enumerate(window_df['residue']):
            pos = i + pad_left
            if res in self.residue_encoder.classes_:
                residues_onehot[pos, self.residue_encoder.transform([res])[0]] = 1.0
        
        for i, ss in enumerate(window_df['ss_type']):
            pos = i + pad_left
            if ss in self.ss_encoder.classes_:
                ss_onehot[pos, self.ss_encoder.transform([ss])[0]] = 1.0
        
        # Chemical shifts - BERT-style masking
        for i, col in enumerate(self.cs_cols):
            values = window_df[col].values
            available = (values != 9999.0)
            
            # Normalize
            normalized = torch.tensor([self.normalize(v, col) for v in values], dtype=torch.float32)
            
            # Place in padded tensor
            cs_target[pad_left:pad_left+actual_seq_len, i] = normalized
            
            # BERT-style masking
            if self.training:
                # Random masking for training
                mask_these = available & (np.random.random(actual_seq_len) < self.mask_prob)
            else:
                # Deterministic masking for test
                if self.seed is not None:
                    rng = np.random.RandomState(self.seed + idx)
                    mask_these = available & (rng.random(actual_seq_len) < self.mask_prob)
                else:
                    mask_these = available & (np.random.random(actual_seq_len) < self.mask_prob)
            
            # Apply mask at correct positions
            for j, masked in enumerate(mask_these):
                pos = j + pad_left
                if masked:
                    cs_mask[pos, i] = True
                    cs_input[pos, i] = 0.0
                elif available[j]:
                    cs_input[pos, i] = normalized[j]
        
        # GUARANTEED MASKING: Ensure at least 1 CS is masked at center position
        center_in_window = self.context_window
        if self.guarantee_mask:
            # Check if any CS is masked at center
            if not cs_mask[center_in_window].any():
                # Find available CS at center (must be in actual data, not padding)
                if pad_left <= center_in_window < pad_left + actual_seq_len:
                    window_center_idx = center_in_window - pad_left
                    available_cs = []
                    for cs_idx, col in enumerate(self.cs_cols):
                        value = window_df.iloc[window_center_idx][col]
                        if value != 9999.0:
                            available_cs.append(cs_idx)
                    
                    if available_cs:
                        # Use seeded RNG for test (deterministic but random)
                        if self.training:
                            mask_idx = np.random.choice(available_cs)
                        else:
                            if self.seed is not None:
                                rng = np.random.RandomState(self.seed + idx)
                                mask_idx = rng.choice(available_cs)
                            else:
                                mask_idx = np.random.choice(available_cs)
                        
                        cs_mask[center_in_window, mask_idx] = True
                        cs_input[center_in_window, mask_idx] = 0.0
        
        # If multiple CS masked at center, randomly pick ONE to predict
        masked_count = cs_mask[center_in_window].sum().item()
        if masked_count > 1:
            # Get indices of all masked CS at center
            masked_indices = torch.where(cs_mask[center_in_window])[0].cpu().numpy()
            
            # Randomly select one to keep masked
            if self.training:
                keep_idx = np.random.choice(masked_indices)
            else:
                if self.seed is not None:
                    rng = np.random.RandomState(self.seed + idx + 10000)
                    keep_idx = rng.choice(masked_indices)
                else:
                    keep_idx = np.random.choice(masked_indices)
            
            # Unmask all others (restore their input values)
            for idx_to_unmask in masked_indices:
                if idx_to_unmask != keep_idx:
                    cs_mask[center_in_window, idx_to_unmask] = False
                    cs_input[center_in_window, idx_to_unmask] = cs_target[center_in_window, idx_to_unmask]
        
        # Mystery feature and angles (fill in actual data positions)
        mystery_values = window_df['mystery_feature'].values
        mystery[pad_left:pad_left+actual_seq_len] = torch.tensor([
            self.normalize(v, 'mystery_feature') for v in mystery_values
        ], dtype=torch.float32)
        
        phi_values = window_df['phi'].values
        phi[pad_left:pad_left+actual_seq_len] = torch.tensor([
            self.normalize(v, 'phi') for v in phi_values
        ], dtype=torch.float32)
        
        psi_values = window_df['psi'].values
        psi[pad_left:pad_left+actual_seq_len] = torch.tensor([
            self.normalize(v, 'psi') for v in psi_values
        ], dtype=torch.float32)
        
        # Concatenate all features: [seq_len, n_residues + n_ss + 4_CS + 3_other]
        features = torch.cat([
            residues_onehot, ss_onehot, cs_input,
            mystery.unsqueeze(-1), phi.unsqueeze(-1), psi.unsqueeze(-1)
        ], dim=-1)
        
        # Only return center residue's target
        cs_target_center = cs_target[center_in_window]
        cs_mask_center = cs_mask[center_in_window]
        
        return {
            'features': features,  # [seq_len, feat_dim]
            'cs_target': cs_target_center,  # [4]
            'cs_mask': cs_mask_center  # [4]
        }

# ============================================================================
# Model - BIGGER with more layers
# ============================================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


class ChemicalShiftCNN(nn.Module):
    def __init__(self, in_feat, channels=[128, 256, 512, 768], kernel=3, dropout=0.3):
        super().__init__()
        
        # Input normalization to fix scale mismatch
        self.input_norm = nn.LayerNorm(in_feat)
        
        # Conv layers - BIGGER
        layers = []
        in_ch = in_feat
        for out_ch in channels:
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        # Attention pooling - increased capacity
        self.attn = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Regression heads for 4 chemical shifts - increased capacity
        self.cs_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels[-1], 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            ) for _ in range(4)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feat)
        Returns:
            predictions: (batch, 4)
        """
        # Normalize input features first
        x = self.input_norm(x)
        
        # Conv expects (batch, feat, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # back to (batch, seq_len, channels)
        
        # Attention pooling
        attn_w = F.softmax(self.attn(x), dim=1)
        x = torch.sum(x * attn_w, dim=1)  # (batch, channels)
        
        # Predict each CS
        predictions = []
        for head in self.cs_heads:
            predictions.append(head(x).squeeze(-1))
        
        predictions = torch.stack(predictions, dim=-1)  # (batch, 4)
        
        return predictions

# ============================================================================
# Training with Wandb Logging
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_count = 0
    batch_times = []
    grad_norms = []
    
    # Track per-CS losses
    cs_losses = {f'CS{i+1}': 0 for i in range(4)}
    cs_counts = {f'CS{i+1}': 0 for i in range(4)}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
        features = batch['features'].to(device)
        cs_target = batch['cs_target'].to(device)
        cs_mask = batch['cs_mask'].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(features)
        
        if cs_mask.sum() == 0:
            continue
            
        loss = F.mse_loss(predictions[cs_mask], cs_target[cs_mask])
        
        # Track per-CS losses
        for i in range(4):
            mask_i = cs_mask[:, i]
            if mask_i.sum() > 0:
                cs_loss = F.mse_loss(predictions[mask_i, i], cs_target[mask_i, i])
                cs_losses[f'CS{i+1}'] += cs_loss.item() * mask_i.sum().item()
                cs_counts[f'CS{i+1}'] += mask_i.sum().item()
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(grad_norm.item())
        optimizer.step()
        
        batch_loss = loss.item()
        batch_count = cs_mask.sum().item()
        total_loss += batch_loss * batch_count
        total_count += batch_count
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        samples_per_sec = len(features) / batch_time if batch_time > 0 else 0
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'grad': f'{grad_norm:.3f}',
            'samples/s': f'{samples_per_sec:.0f}'
        })
    
    epoch_time = time.time() - epoch_start
    
    # Calculate per-CS average losses
    avg_cs_losses = {k: (v / cs_counts[k] if cs_counts[k] > 0 else 0) 
                     for k, v in cs_losses.items()}
    
    return {
        'loss': total_loss / total_count if total_count > 0 else 0,
        'epoch_time': epoch_time,
        'samples_per_sec': len(loader.dataset) / epoch_time,
        'avg_batch_time': np.mean(batch_times),
        'cs_losses': avg_cs_losses,
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_std': np.std(grad_norms),
        'grad_norm_max': np.max(grad_norms)
    }

def evaluate(model, loader, device, stats, cs_cols, epoch=None, log_to_wandb=True):
    model.eval()
    total_loss = 0
    total_count = 0
    per_cs_errors = {col: [] for col in cs_cols}
    
    # Collect detailed statistics
    all_predictions = []
    all_targets = []
    all_masks = []
    all_residues = []
    all_ss_types = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            features = batch['features'].to(device)
            cs_target = batch['cs_target'].to(device)
            cs_mask = batch['cs_mask'].to(device)
            
            predictions = model(features)
            
            if cs_mask.sum() == 0:
                continue
                
            loss = F.mse_loss(predictions[cs_mask], cs_target[cs_mask])
            
            batch_count = cs_mask.sum().item()
            total_loss += loss.item() * batch_count
            total_count += batch_count
            
            # Collect for detailed analysis
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(cs_target.cpu().numpy())
            all_masks.append(cs_mask.cpu().numpy())
            
            # Per-CS type errors
            for i, col in enumerate(cs_cols):
                mask_i = cs_mask[:, i]
                if mask_i.sum() > 0:
                    pred_i = predictions[:, i][mask_i]
                    true_i = cs_target[:, i][mask_i]
                    
                    # Denormalize
                    pred_i = pred_i * stats[col]['std'] + stats[col]['mean']
                    true_i = true_i * stats[col]['std'] + stats[col]['mean']
                    
                    mae = torch.abs(pred_i - true_i).mean().item()
                    per_cs_errors[col].append(mae)
    
    # Concatenate all results
    if len(all_predictions) == 0:
        print("WARNING: No predictions collected during evaluation!")
        print("This can happen if all samples have missing CS values.")
        return 0.0, {}
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Denormalize for analysis
    predictions_denorm = np.zeros_like(all_predictions)
    targets_denorm = np.zeros_like(all_targets)
    
    for i, col in enumerate(cs_cols):
        predictions_denorm[:, i] = all_predictions[:, i] * stats[col]['std'] + stats[col]['mean']
        targets_denorm[:, i] = all_targets[:, i] * stats[col]['std'] + stats[col]['mean']
    
    errors_denorm = np.abs(predictions_denorm - targets_denorm)
    
    avg_loss = total_loss / total_count if total_count > 0 else 0
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print(f"  Overall Loss (normalized): {avg_loss:.4f}")
    print(f"  MAE per chemical shift type (original scale):")
    
    mae_results = {}
    for col in cs_cols:
        if per_cs_errors[col]:
            mae = np.mean(per_cs_errors[col])
            mae_results[col] = mae
            print(f"    {col:8s}: {mae:.3f} ppm")
    print("="*60 + "\n")
    
    # Detailed wandb logging
    if log_to_wandb and USE_WANDB and epoch is not None:
        log_dict = {}
        
        # Per-CS detailed statistics
        for i, col in enumerate(cs_cols):
            mask_i = all_masks[:, i]
            if mask_i.sum() > 0:
                errors_i = errors_denorm[mask_i, i]
                
                log_dict[f'eval_detail/{col}_mae'] = np.mean(errors_i)
                log_dict[f'eval_detail/{col}_median'] = np.median(errors_i)
                log_dict[f'eval_detail/{col}_std'] = np.std(errors_i)
                log_dict[f'eval_detail/{col}_p95'] = np.percentile(errors_i, 95)
                log_dict[f'eval_detail/{col}_max'] = np.max(errors_i)
                
                # Error distribution histogram
                log_dict[f'eval_hist/{col}_errors'] = wandb.Histogram(errors_i)
                
                # Prediction scatter (sample)
                pred_i = predictions_denorm[mask_i, i]
                target_i = targets_denorm[mask_i, i]
                if len(pred_i) > 1000:
                    sample_idx = np.random.choice(len(pred_i), 1000, replace=False)
                    pred_i = pred_i[sample_idx]
                    target_i = target_i[sample_idx]
                
                # Create scatter plot data
                scatter_data = [[t, p] for t, p in zip(target_i, pred_i)]
                table = wandb.Table(data=scatter_data, columns=["true", "predicted"])
                log_dict[f'eval_scatter/{col}'] = wandb.plot.scatter(
                    table, "true", "predicted",
                    title=f"{col} Predictions vs True"
                )
        
        # Overall statistics
        log_dict['eval_detail/total_predictions'] = all_masks.sum()
        log_dict['eval_detail/avg_mae_all_cs'] = np.mean([mae_results[col] for col in cs_cols if col in mae_results])
        
        # Best and worst predictions
        for i, col in enumerate(cs_cols):
            mask_i = all_masks[:, i]
            if mask_i.sum() > 0:
                errors_i = errors_denorm[mask_i, i]
                log_dict[f'eval_detail/{col}_best_10_mean'] = np.mean(np.sort(errors_i)[:10])
                log_dict[f'eval_detail/{col}_worst_10_mean'] = np.mean(np.sort(errors_i)[-10:])
        
        wandb.log(log_dict, step=epoch)
    
    return avg_loss, mae_results

# ============================================================================
# Training Statistics Visualization
# ============================================================================

def plot_training_statistics(history, save_path='training_stats.png'):
    """Create comprehensive training visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Overall loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (normalized)', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning rate schedule
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, history['learning_rate'], color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-CS MAE
    ax3 = fig.add_subplot(gs[1, :])
    for cs in ['CS1', 'CS2', 'CS3', 'CS4']:
        if cs in history['test_mae']:
            ax3.plot(epochs, history['test_mae'][cs], label=cs, linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('MAE (ppm)', fontsize=12)
    ax3.set_title('Test MAE by Chemical Shift Type', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Training speed
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(epochs, history['samples_per_sec'], color='purple', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Samples/sec', fontsize=12)
    ax4.set_title('Training Speed', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Gradient norms (if available)
    if 'grad_norm' in history:
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(epochs, history['grad_norm'], color='orange', linewidth=2)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Gradient Norm', fontsize=12)
        ax5.set_title('Gradient Norm', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Best performance summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    best_epoch = np.argmin(history['test_loss']) + 1
    best_test_loss = min(history['test_loss'])
    
    summary_text = "Best Performance:\n\n"
    summary_text += f"Epoch: {best_epoch}\n"
    summary_text += f"Test Loss: {best_test_loss:.4f}\n\n"
    summary_text += "Best MAE (ppm):\n"
    
    for cs in ['CS1', 'CS2', 'CS3', 'CS4']:
        if cs in history['test_mae']:
            best_mae = min(history['test_mae'][cs])
            summary_text += f"  {cs}: {best_mae:.3f}\n"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Training statistics saved to {save_path}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("IMPROVED CHEMICAL SHIFT PREDICTION - Training with WandB")
    print("WITH EDGE RESIDUE MASKING")
    print("USING TRAINTEST.CSV (90% holdout, eval.csv reserved for final testing)")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Masking probability: {MASK_PROB}")
    print(f"Context window: ±{CONTEXT_WINDOW//2} residues")
    print(f"Architecture: 1D CNN with {len(CONV_CHANNELS)} residual blocks")
    print(f"Channels: {CONV_CHANNELS}")
    print(f"Input normalization: ENABLED (LayerNorm)")
    print(f"Guaranteed masking: ENABLED for train AND test")
    print(f"Edge residues: INCLUDED with masked context")
    
    # Initialize wandb
    if USE_WANDB:
        # Login to wandb (you can also run 'wandb login' once in terminal instead)
        # wandb.login(key="YOUR_API_KEY_HERE")  # Uncomment and add your key, or use env var
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "mask_prob": MASK_PROB,
                "context_window": CONTEXT_WINDOW,
                "conv_channels": CONV_CHANNELS,
                "dropout": DROPOUT,
                "device": DEVICE,
                "architecture": "ResidualCNN",
                "guaranteed_masking": True,
                "edge_masking": True,
                "data_source": "traintest.csv (90% holdout)"
            }
        )
    
    # Load data
    train_df, test_df = load_and_split_data(DATA_PATH)
    stats = compute_stats(train_df)
    
    # Create datasets with improvements
    train_dataset = ProteinDataset(
        train_df, stats, context_window=CONTEXT_WINDOW//2, 
        mask_prob=MASK_PROB, training=True, verbose=True,
        guarantee_mask=True
    )
    
    print(f"\nTrain residue types: {train_dataset.n_residue_types}")
    print(f"Train SS types: {train_dataset.n_ss_types}")
    
    test_dataset = ProteinDataset(
        test_df, stats, context_window=CONTEXT_WINDOW//2, 
        mask_prob=MASK_PROB, training=False, verbose=False,
        residue_encoder=train_dataset.residue_encoder,
        ss_encoder=train_dataset.ss_encoder,
        seed=42,
        guarantee_mask=True
    )
    
    print(f"Test residue types: {test_dataset.n_residue_types}")
    print(f"Test SS types: {test_dataset.n_ss_types}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=16, pin_memory=True if DEVICE in ['cuda', 'mps'] else False,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=16, pin_memory=True if DEVICE in ['cuda', 'mps'] else False,
        persistent_workers=True, prefetch_factor=4
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Calculate input feature dimension
    in_feat = train_dataset.n_residue_types + train_dataset.n_ss_types + 4 + 3
    print(f"Input feature dimension: {in_feat}")
    
    # Create model
    model = ChemicalShiftCNN(
        in_feat=in_feat, channels=CONV_CHANNELS, kernel=KERNEL_SIZE, dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    if USE_WANDB:
        wandb.watch(model, log='all', log_freq=100)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rate': [],
        'samples_per_sec': [],
        'grad_norm': [],
        'test_mae': {'CS1': [], 'CS2': [], 'CS3': [], 'CS4': []}
    }
    
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_stats = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Time: {train_stats['epoch_time']:.1f}s")
        print(f"  Throughput: {train_stats['samples_per_sec']:.0f} samples/s")
        
        # Evaluate
        test_loss, mae_results = evaluate(model, test_loader, DEVICE, stats, train_dataset.cs_cols, 
                                          epoch=epoch, log_to_wandb=USE_WANDB)
        
        # Update scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Log GroupNorm parameters to wandb
        if USE_WANDB and epoch % 5 == 0:
            gn_stats = {}
            for name, param in model.named_parameters():
                if 'gn' in name and 'weight' in name:
                    gn_stats[f'params/{name}_mean'] = param.mean().item()
                    gn_stats[f'params/{name}_std'] = param.std().item()
                    gn_stats[f'params/{name}_min'] = param.min().item()
                    gn_stats[f'params/{name}_max'] = param.max().item()
            
            if gn_stats:
                wandb.log(gn_stats, step=epoch)
        
        # Update history
        history['train_loss'].append(train_stats['loss'])
        history['test_loss'].append(test_loss)
        history['learning_rate'].append(current_lr)
        history['samples_per_sec'].append(train_stats['samples_per_sec'])
        history['grad_norm'].append(train_stats['grad_norm_mean'])
        for cs, mae in mae_results.items():
            history['test_mae'][cs].append(mae)
        
        # Log to wandb
        if USE_WANDB:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/samples_per_sec': train_stats['samples_per_sec'],
                'train/grad_norm_mean': train_stats['grad_norm_mean'],
                'train/grad_norm_std': train_stats['grad_norm_std'],
                'train/grad_norm_max': train_stats['grad_norm_max'],
                'test/loss': test_loss,
                'learning_rate': current_lr
            }
            
            # Log per-CS losses
            for cs, loss in train_stats['cs_losses'].items():
                log_dict[f'train/loss_{cs}'] = loss
            
            for cs, mae in mae_results.items():
                log_dict[f'test/mae_{cs}'] = mae
            
            wandb.log(log_dict)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_improved.pt')
            print(f"  *** New best model saved! Test Loss: {test_loss:.4f} ***")
            
            # Save best MAE results
            with open('best_mae_results.json', 'w') as f:
                json.dump(mae_results, f, indent=2)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print("="*80)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate training statistics plots
    plot_training_statistics(history, 'training_stats.png')
    
    if USE_WANDB:
        # Log final plot to wandb
        wandb.log({"training_statistics": wandb.Image('training_stats.png')})
        wandb.finish()
    
    print("\n✓ Training history saved to training_history.json")
    print("✓ Training statistics plot saved to training_stats.png")
    print("✓ Best model saved to best_model_improved.pt")
    print("\n" + "="*80)
    print("NOTE: eval.csv (10% holdout) remains untouched for final evaluation")
    print("="*80)

if __name__ == '__main__':
    main()