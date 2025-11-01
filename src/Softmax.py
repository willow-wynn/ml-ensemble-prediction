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

# Set number of threads to use all CPU cores
torch.set_num_threads(os.cpu_count())

# Configuration
DATA_PATH = '/Users/wynndiaz/he_lab/data/improved.csv'
BATCH_SIZE = 32  # Increased from 16
MAX_SEQ_LEN = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model hyperparameters (kept small to avoid overfitting)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

print(f"Using device: {DEVICE}")

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_split_data(csv_path, test_size=0.15, random_state=42):
    """Load data and split by protein ID"""
    print("Loading data...")
    
    # Read CSV with explicit column types
    df = pd.read_csv(
        csv_path, 
        header=None,
        dtype={
            0: int,    # protein_id
            1: int,    # position
            2: str,    # residue
            3: float,  # CS1
            4: float,  # CS2
            5: float,  # CS3
            6: float,  # CS4
            7: str,    # secondary structure
            8: float,  # mystery feature
            9: float,  # phi
            10: float  # psi
        }
    )
    
    df.columns = ['protein_id', 'position', 'residue', 'CS1', 'CS2', 
                  'CS3', 'CS4', 'ss_type', 'mystery_feature', 'phi', 'psi']
    
    print(f"Total rows: {len(df):,}")
    print(f"Total proteins: {df['protein_id'].nunique():,}")
    
    # Get unique proteins and split
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
    
    # Mystery feature
    stats['mystery_feature'] = {
        'mean': float(df['mystery_feature'].mean()),
        'std': float(df['mystery_feature'].std())
    }
    print(f"{'mystery_feature':15s}: mean={stats['mystery_feature']['mean']:8.2f}, std={stats['mystery_feature']['std']:7.2f}")
    
    # Angles
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
    def __init__(self, df, stats, max_seq_len=256):
        self.df = df
        self.stats = stats
        self.max_seq_len = max_seq_len
        self.cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
        
        # Encode categorical features
        self.residue_encoder = LabelEncoder()
        self.ss_encoder = LabelEncoder()
        
        all_residues = df['residue'].unique()
        all_ss = df['ss_type'].unique()
        self.residue_encoder.fit(all_residues)
        self.ss_encoder.fit(all_ss)
        
        self.n_residue_types = len(self.residue_encoder.classes_)
        self.n_ss_types = len(self.ss_encoder.classes_)
        
        print(f"\nResidue types: {self.n_residue_types} ({list(self.residue_encoder.classes_)[:5]}...)")
        print(f"SS types: {self.n_ss_types} ({list(self.ss_encoder.classes_)})")
        
        # Group by protein
        self.proteins = list(df.groupby('protein_id'))
        
    def normalize(self, value, col):
        if value == 9999.0 or np.isnan(value):
            return 0.0
        return (value - self.stats[col]['mean']) / (self.stats[col]['std'] + 1e-8)
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_id, prot_df = self.proteins[idx]
        prot_df = prot_df.sort_values('position').reset_index(drop=True)
        
        seq_len = min(len(prot_df), self.max_seq_len)
        prot_df = prot_df.iloc[:seq_len]
        
        # Encode features
        residues = torch.tensor(
            self.residue_encoder.transform(prot_df['residue']), dtype=torch.long
        )
        ss_types = torch.tensor(
            self.ss_encoder.transform(prot_df['ss_type']), dtype=torch.long
        )
        
        # Chemical shifts and masks (4 shifts now!)
        cs_values = torch.zeros(seq_len, 4, dtype=torch.float32)
        cs_mask = torch.zeros(seq_len, 4, dtype=torch.bool)
        
        for i, col in enumerate(self.cs_cols):
            values = prot_df[col].values
            cs_mask[:, i] = torch.tensor(values != 9999.0)
            cs_values[:, i] = torch.tensor([self.normalize(v, col) for v in values], dtype=torch.float32)
        
        # Mystery feature (normalized)
        mystery = torch.tensor([
            self.normalize(v, 'mystery_feature') for v in prot_df['mystery_feature'].values
        ], dtype=torch.float32)
        
        # Angles
        phi = torch.tensor([self.normalize(v, 'phi') for v in prot_df['phi'].values], dtype=torch.float32)
        psi = torch.tensor([self.normalize(v, 'psi') for v in prot_df['psi'].values], dtype=torch.float32)
        phi_mask = torch.tensor(prot_df['phi'].values != 9999.0, dtype=torch.bool)
        psi_mask = torch.tensor(prot_df['psi'].values != 9999.0, dtype=torch.bool)
        
        # Padding
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            residues = F.pad(residues, (0, pad_len), value=0)
            ss_types = F.pad(ss_types, (0, pad_len), value=0)
            cs_values = F.pad(cs_values, (0, 0, 0, pad_len), value=0)
            cs_mask = F.pad(cs_mask, (0, 0, 0, pad_len), value=False)
            mystery = F.pad(mystery, (0, pad_len), value=0)
            phi = F.pad(phi, (0, pad_len), value=0)
            psi = F.pad(psi, (0, pad_len), value=0)
            phi_mask = F.pad(phi_mask, (0, pad_len), value=False)
            psi_mask = F.pad(psi_mask, (0, pad_len), value=False)
        
        # Attention mask (for padding)
        attn_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attn_mask[:seq_len] = True
        
        return {
            'residues': residues,
            'ss_types': ss_types,
            'cs_values': cs_values,
            'cs_mask': cs_mask,
            'mystery': mystery,
            'phi': phi,
            'psi': psi,
            'phi_mask': phi_mask,
            'psi_mask': psi_mask,
            'attn_mask': attn_mask,
            'seq_len': seq_len
        }

# ============================================================================
# Model
# ============================================================================

class ChemicalShiftTransformer(nn.Module):
    def __init__(self, n_residue_types, n_ss_types, d_model=128, n_heads=4, 
                 n_layers=3, dim_feedforward=256, dropout=0.1, max_seq_len=256):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.residue_embed = nn.Embedding(n_residue_types, d_model // 4)
        self.ss_embed = nn.Embedding(n_ss_types, d_model // 4)
        self.pos_embed = nn.Embedding(max_seq_len, d_model // 4)
        
        # Input projection (embeddings + mystery + angles -> d_model)
        # d_model//4 * 3 + 3 continuous features (mystery, phi, psi)
        self.input_proj = nn.Linear(d_model // 4 * 3 + 3, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads for each chemical shift type (4 now!)
        self.cs_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(4)
        ])
        
    def forward(self, residues, ss_types, mystery, phi, psi, attn_mask):
        batch_size, seq_len = residues.shape
        
        # Create embeddings
        pos = torch.arange(seq_len, device=residues.device).unsqueeze(0).expand(batch_size, -1)
        
        res_emb = self.residue_embed(residues)
        ss_emb = self.ss_embed(ss_types)
        pos_emb = self.pos_embed(pos)
        
        # Concatenate all features
        x = torch.cat([
            res_emb, ss_emb, pos_emb, 
            mystery.unsqueeze(-1), phi.unsqueeze(-1), psi.unsqueeze(-1)
        ], dim=-1)
        x = self.input_proj(x)
        
        # Create padding mask for transformer (True = ignore)
        padding_mask = ~attn_mask
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Predict each chemical shift
        cs_predictions = []
        for head in self.cs_heads:
            cs_predictions.append(head(x).squeeze(-1))
        
        # Stack: [batch, seq_len, 4]
        cs_predictions = torch.stack(cs_predictions, dim=-1)
        
        return cs_predictions

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_count = 0
    batch_times = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    log_interval = max(1, len(loader) // 4)  # Log 4 times per epoch
    
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
        residues = batch['residues'].to(device)
        ss_types = batch['ss_types'].to(device)
        mystery = batch['mystery'].to(device)
        phi = batch['phi'].to(device)
        psi = batch['psi'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        cs_values = batch['cs_values'].to(device)
        cs_mask = batch['cs_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(residues, ss_types, mystery, phi, psi, attn_mask)
        
        # Compute loss only on non-missing values
        loss = F.mse_loss(predictions[cs_mask], cs_values[cs_mask])
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        batch_loss = loss.item()
        batch_count = cs_mask.sum().item()
        total_loss += batch_loss * batch_count
        total_count += batch_count
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calculate throughput
        samples_per_sec = len(batch['residues']) / batch_time if batch_time > 0 else 0
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'grad': f'{grad_norm:.3f}',
            'samples/s': f'{samples_per_sec:.0f}'
        })
        
        # Mid-epoch logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_count
            avg_time = np.mean(batch_times[-log_interval:])
            avg_throughput = BATCH_SIZE / avg_time if avg_time > 0 else 0
            print(f"\n  Batch {batch_idx+1}/{len(loader)} - Loss: {avg_loss:.4f} | "
                  f"GradNorm: {grad_norm:.4f} | {avg_throughput:.0f} samples/s")
    
    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / total_count,
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
            residues = batch['residues'].to(device)
            ss_types = batch['ss_types'].to(device)
            mystery = batch['mystery'].to(device)
            phi = batch['phi'].to(device)
            psi = batch['psi'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            cs_values = batch['cs_values'].to(device)
            cs_mask = batch['cs_mask'].to(device)
            
            predictions = model(residues, ss_types, mystery, phi, psi, attn_mask)
            
            loss = F.mse_loss(predictions[cs_mask], cs_values[cs_mask])
            
            batch_count = cs_mask.sum().item()
            total_loss += loss.item() * batch_count
            total_count += batch_count
            
            # Per-CS type errors (in original scale)
            for i, col in enumerate(cs_cols):
                mask_i = cs_mask[:, :, i]
                if mask_i.sum() > 0:
                    pred_i = predictions[:, :, i][mask_i]
                    true_i = cs_values[:, :, i][mask_i]
                    
                    # Denormalize
                    pred_i = pred_i * stats[col]['std'] + stats[col]['mean']
                    true_i = true_i * stats[col]['std'] + stats[col]['mean']
                    
                    mae = torch.abs(pred_i - true_i).mean().item()
                    per_cs_errors[col].append(mae)
    
    avg_loss = total_loss / total_count
    
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
    # Load and split data
    train_df, test_df = load_and_split_data(DATA_PATH)
    
    # Compute statistics
    stats = compute_stats(train_df)
    
    # Create datasets
    train_dataset = ProteinDataset(train_df, stats, max_seq_len=MAX_SEQ_LEN)
    test_dataset = ProteinDataset(test_df, stats, max_seq_len=MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True if DEVICE in ['cuda', 'mps'] else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if DEVICE in ['cuda', 'mps'] else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Create model
    model = ChemicalShiftTransformer(
        n_residue_types=train_dataset.n_residue_types,
        n_ss_types=train_dataset.n_ss_types,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_test_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        train_stats = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Time: {train_stats['epoch_time']:.1f}s")
        print(f"  Throughput: {train_stats['samples_per_sec']:.0f} samples/s")
        print(f"  Avg batch time: {train_stats['avg_batch_time']*1000:.1f}ms")
        
        # Evaluate on test set
        test_loss = evaluate(model, test_loader, DEVICE, stats, train_dataset.cs_cols)
        
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  *** New best model saved! Test Loss: {test_loss:.4f} ***")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()