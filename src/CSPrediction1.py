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

# Enable MPS fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set number of threads to use all CPU cores
torch.set_num_threads(os.cpu_count())

# Configuration
DATA_PATH = '/Users/wynndiaz/he_lab/data/improved.csv'
BATCH_SIZE = 256  # Larger batch for better generalization
MAX_SEQ_LEN = 256
EPOCHS = 250
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MASK_PROB = 0.2  # Mask 20% of known values

# Model hyperparameters - bigger model
CONTEXT_WINDOW = 9  # ±3 residues
CONV_CHANNELS = [128, 256, 512]  # Increased from [64, 128, 256]
KERNEL_SIZE = 3
DROPOUT = 0.3  # Higher dropout

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
# Dataset
# ============================================================================

class ProteinDataset(Dataset):
    def __init__(self, df, stats, context_window=3, mask_prob=0.2, training=True, 
                 verbose=True, residue_encoder=None, ss_encoder=None, seed=None):
        self.df = df
        self.stats = stats
        self.context_window = context_window
        self.mask_prob = mask_prob
        self.training = training
        self.seed = seed  # Fixed seed for deterministic masking in test
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
        
        # Build valid indices (need context window on both sides AND contiguous positions)
        self.samples = []
        for protein_id, prot_df in self.proteins:
            prot_df = prot_df.sort_values('position').reset_index(drop=True)
            for i in range(context_window, len(prot_df) - context_window):
                # Check if positions are contiguous (no gaps)
                window_start = i - context_window
                window_end = i + context_window
                
                start_pos = prot_df.iloc[window_start]['position']
                end_pos = prot_df.iloc[window_end]['position']
                expected_span = 2 * context_window  # Should span exactly this many positions
                actual_span = end_pos - start_pos
                
                # Only add if positions are contiguous
                if actual_span == expected_span:
                    self.samples.append((protein_id, i))
        
        if verbose:
            print(f"Valid samples with context: {len(self.samples):,}")
        
    def normalize(self, value, col):
        if value == 9999.0 or np.isnan(value):
            return 0.0
        return (value - self.stats[col]['mean']) / (self.stats[col]['std'] + 1e-8)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        protein_id, center_idx = self.samples[idx]
        prot_df = self.df[self.df['protein_id'] == protein_id].sort_values('position').reset_index(drop=True)
        
        # Extract context window
        start_idx = center_idx - self.context_window
        end_idx = center_idx + self.context_window + 1
        window_df = prot_df.iloc[start_idx:end_idx]
        
        seq_len = len(window_df)
        
        # Encode residues (one-hot, 20 AAs)
        residues_onehot = torch.zeros(seq_len, self.n_residue_types, dtype=torch.float32)
        for i, res in enumerate(window_df['residue']):
            if res in self.residue_encoder.classes_:
                residues_onehot[i, self.residue_encoder.transform([res])[0]] = 1.0
        
        # Encode SS (one-hot)
        ss_onehot = torch.zeros(seq_len, self.n_ss_types, dtype=torch.float32)
        for i, ss in enumerate(window_df['ss_type']):
            if ss in self.ss_encoder.classes_:
                ss_onehot[i, self.ss_encoder.transform([ss])[0]] = 1.0
        
        # Chemical shifts - BERT-style masking
        cs_input = torch.zeros(seq_len, 4, dtype=torch.float32)
        cs_target = torch.zeros(seq_len, 4, dtype=torch.float32)
        cs_mask = torch.zeros(seq_len, 4, dtype=torch.bool)
        
        for i, col in enumerate(self.cs_cols):
            values = window_df[col].values
            available = (values != 9999.0)
            
            # Normalize
            normalized = torch.tensor([self.normalize(v, col) for v in values], dtype=torch.float32)
            cs_target[:, i] = normalized
            
            # BERT-style masking
            if self.training:
                # Random masking for training
                mask_these = available & (np.random.random(seq_len) < self.mask_prob)
                cs_mask[:, i] = torch.tensor(mask_these)
                mask_these_tensor = torch.tensor(mask_these, dtype=torch.bool)
                cs_input[:, i] = normalized * (~mask_these_tensor).float()
            else:
                # Deterministic masking for test (same positions every time)
                if self.seed is not None:
                    # Use sample index as seed for reproducible masking
                    rng = np.random.RandomState(self.seed + idx)
                    mask_these = available & (rng.random(seq_len) < self.mask_prob)
                else:
                    mask_these = available & (np.random.random(seq_len) < self.mask_prob)
                
                cs_mask[:, i] = torch.tensor(mask_these)
                mask_these_tensor = torch.tensor(mask_these, dtype=torch.bool)
                cs_input[:, i] = normalized * (~mask_these_tensor).float()
        
        # Mystery feature and angles
        mystery = torch.tensor([
            self.normalize(v, 'mystery_feature') for v in window_df['mystery_feature'].values
        ], dtype=torch.float32)
        
        phi = torch.tensor([self.normalize(v, 'phi') for v in window_df['phi'].values], dtype=torch.float32)
        psi = torch.tensor([self.normalize(v, 'psi') for v in window_df['psi'].values], dtype=torch.float32)
        
        # Concatenate all features: [seq_len, n_residues + n_ss + 4_CS + 3_other]
        features = torch.cat([
            residues_onehot, ss_onehot, cs_input,
            mystery.unsqueeze(-1), phi.unsqueeze(-1), psi.unsqueeze(-1)
        ], dim=-1)
        
        # Only return center residue's target
        center_in_window = self.context_window
        cs_target_center = cs_target[center_in_window]
        cs_mask_center = cs_mask[center_in_window]
        
        return {
            'features': features,  # [seq_len, feat_dim]
            'cs_target': cs_target_center,  # [4]
            'cs_mask': cs_mask_center  # [4]
        }

# ============================================================================
# Model
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
    def __init__(self, in_feat, channels=[128, 256, 512], kernel=3, dropout=0.3):
        super().__init__()
        
        # Conv layers
        layers = []
        in_ch = in_feat
        for out_ch in channels:
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        # Attention pooling - increased capacity
        self.attn = nn.Sequential(
            nn.Linear(channels[-1], 128),  # Increased from 64
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Regression heads for 4 chemical shifts - increased capacity
        self.cs_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels[-1], 256),  # Increased from 128
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),  # Increased from 64
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
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_count = 0
    batch_times = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    log_interval = max(1, len(loader) // 4)
    
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
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_count if total_count > 0 else 0
            avg_time = np.mean(batch_times[-log_interval:])
            avg_throughput = BATCH_SIZE / avg_time if avg_time > 0 else 0
            print(f"\n  Batch {batch_idx+1}/{len(loader)} - Loss: {avg_loss:.4f} | "
                  f"GradNorm: {grad_norm:.4f} | {avg_throughput:.0f} samples/s")
    
    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / total_count if total_count > 0 else 0,
        'epoch_time': epoch_time,
        'samples_per_sec': len(loader.dataset) / epoch_time,
        'avg_batch_time': np.mean(batch_times)
    }

def evaluate(model, loader, device, stats, cs_cols):
    model.eval()
    total_loss = 0
    total_count = 0
    per_cs_errors = {col: [] for col in cs_cols}
    
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
    
    avg_loss = total_loss / total_count if total_count > 0 else 0
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print(f"  Overall Loss (normalized): {avg_loss:.4f}")
    print(f"  MAE per chemical shift type (original scale):")
    for col in cs_cols:
        if per_cs_errors[col]:
            mae = np.mean(per_cs_errors[col])
            print(f"    {col:8s}: {mae:.3f} ppm")
    print("="*60 + "\n")
    
    return avg_loss

# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Masking probability: {MASK_PROB}")
    print(f"Context window: ±{CONTEXT_WINDOW//2} residues")
    print(f"Architecture: 1D CNN with residual blocks")
    
    train_df, test_df = load_and_split_data(DATA_PATH)
    stats = compute_stats(train_df)
    
    train_dataset = ProteinDataset(train_df, stats, context_window=CONTEXT_WINDOW//2, 
                                   mask_prob=MASK_PROB, training=True, verbose=True)
    
    print(f"\nTrain residue types: {train_dataset.n_residue_types}")
    print(f"Train SS types: {train_dataset.n_ss_types}")
    
    # Create test dataset using SAME encoders as train and fixed seed for reproducible masking
    test_dataset = ProteinDataset(test_df, stats, context_window=CONTEXT_WINDOW//2, 
                                  mask_prob=MASK_PROB, training=False, verbose=False,
                                  residue_encoder=train_dataset.residue_encoder,
                                  ss_encoder=train_dataset.ss_encoder,
                                  seed=42)  # Fixed seed for consistent test masking
    
    print(f"Test residue types: {test_dataset.n_residue_types}")
    print(f"Test SS types: {test_dataset.n_ss_types}")
    
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
    
    model = ChemicalShiftCNN(
        in_feat=in_feat, channels=CONV_CHANNELS, kernel=KERNEL_SIZE, dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        train_stats = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Time: {train_stats['epoch_time']:.1f}s")
        print(f"  Throughput: {train_stats['samples_per_sec']:.0f} samples/s")
        
        test_loss = evaluate(model, test_loader, DEVICE, stats, train_dataset.cs_cols)
        
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_cnn.pt')
            print(f"  *** New best model saved! Test Loss: {test_loss:.4f} ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()