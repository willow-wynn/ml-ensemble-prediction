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
BATCH_SIZE = 128
MAX_SEQ_LEN = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = 'cpu'
MASK_PROB = 0.15  # Mask 15% of known values (BERT-style)

# Model hyperparameters
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.5

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_split_data(csv_path, test_size=0.15, random_state=42):
    """Load data and split by protein ID"""
    print("Loading data...")
    
    # Read CSV and let pandas infer types, skip bad lines
    df = pd.read_csv(
        csv_path, 
        header=None,
        on_bad_lines='skip'
    )
    
    print(f"Loaded {len(df)} rows")
    
    # Check column count
    if len(df.columns) != 11:
        raise ValueError(f"Expected 11 columns but got {len(df.columns)}")
    
    df.columns = ['protein_id', 'position', 'residue', 'CS1', 'CS2', 
                  'CS3', 'CS4', 'ss_type', 'mystery_feature', 'phi', 'psi']
    
    # Convert types with error handling
    df['protein_id'] = pd.to_numeric(df['protein_id'], errors='coerce').astype('Int64')
    df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')
    
    # Drop rows with invalid protein_id or position
    df = df.dropna(subset=['protein_id', 'position'])
    
    # Convert numeric columns
    for col in ['CS1', 'CS2', 'CS3', 'CS4', 'mystery_feature', 'phi', 'psi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining bad rows
    before = len(df)
    df = df.dropna()
    if before > len(df):
        print(f"Dropped {before - len(df)} rows with unparseable data")
    
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
    def __init__(self, df, stats, max_seq_len=256, mask_prob=0.15, training=True, verbose=True):
        self.df = df
        self.stats = stats
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.training = training
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
        
        if verbose:
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
        
        # Chemical shifts - THESE ARE NOW INPUT FEATURES!
        cs_values_input = torch.zeros(seq_len, 4, dtype=torch.float32)  # What model sees
        cs_values_target = torch.zeros(seq_len, 4, dtype=torch.float32)  # Ground truth
        cs_available = torch.zeros(seq_len, 4, dtype=torch.bool)  # Which values exist (not 9999)
        cs_mask_target = torch.zeros(seq_len, 4, dtype=torch.bool)  # Which to predict (BERT-style)
        
        for i, col in enumerate(self.cs_cols):
            values = prot_df[col].values
            available = (values != 9999.0)
            cs_available[:, i] = torch.tensor(available)
            
            # Normalize all values
            normalized = torch.tensor([self.normalize(v, col) for v in values], dtype=torch.float32)
            cs_values_target[:, i] = normalized
            
            # BERT-style masking: mask some available values during training
            if self.training:
                # Random mask for available values
                mask_these = available & (np.random.random(seq_len) < self.mask_prob)
                cs_mask_target[:, i] = torch.tensor(mask_these)
                
                # Input: use value if not masked, else 0
                mask_these_tensor = torch.tensor(mask_these, dtype=torch.bool)
                cs_values_input[:, i] = normalized * (~mask_these_tensor).float()
            else:
                # At test time: mask all available values (predict them all)
                cs_mask_target[:, i] = torch.tensor(available)
                cs_values_input[:, i] = torch.zeros(seq_len, dtype=torch.float32)  # Mask all
        
        # Mystery feature (normalized)
        mystery = torch.tensor([
            self.normalize(v, 'mystery_feature') for v in prot_df['mystery_feature'].values
        ], dtype=torch.float32)
        
        # Angles
        phi = torch.tensor([self.normalize(v, 'phi') for v in prot_df['phi'].values], dtype=torch.float32)
        psi = torch.tensor([self.normalize(v, 'psi') for v in prot_df['psi'].values], dtype=torch.float32)
        
        # Padding
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            residues = F.pad(residues, (0, pad_len), value=0)
            ss_types = F.pad(ss_types, (0, pad_len), value=0)
            cs_values_input = F.pad(cs_values_input, (0, 0, 0, pad_len), value=0)
            cs_values_target = F.pad(cs_values_target, (0, 0, 0, pad_len), value=0)
            cs_available = F.pad(cs_available, (0, 0, 0, pad_len), value=False)
            cs_mask_target = F.pad(cs_mask_target, (0, 0, 0, pad_len), value=False)
            mystery = F.pad(mystery, (0, pad_len), value=0)
            phi = F.pad(phi, (0, pad_len), value=0)
            psi = F.pad(psi, (0, pad_len), value=0)
        
        # Attention mask (for padding)
        attn_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attn_mask[:seq_len] = True
        
        return {
            'residues': residues,
            'ss_types': ss_types,
            'cs_input': cs_values_input,  # CS values model can see (masked)
            'cs_target': cs_values_target,  # Ground truth CS values
            'cs_mask': cs_mask_target,  # Which positions to predict
            'mystery': mystery,
            'phi': phi,
            'psi': psi,
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
        
        # Input projection (embeddings + CS values + mystery + angles -> d_model)
        # d_model//4 * 3 + 4 CS values + 3 continuous features (mystery, phi, psi)
        self.input_proj = nn.Linear(d_model // 4 * 3 + 4 + 3, d_model)
        
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
        
    def forward(self, residues, ss_types, cs_input, mystery, phi, psi, attn_mask):
        batch_size, seq_len = residues.shape
        
        # Create embeddings
        pos = torch.arange(seq_len, device=residues.device).unsqueeze(0).expand(batch_size, -1)
        
        res_emb = self.residue_embed(residues)
        ss_emb = self.ss_embed(ss_types)
        pos_emb = self.pos_embed(pos)
        
        # Concatenate all features INCLUDING chemical shifts as input!
        x = torch.cat([
            res_emb, ss_emb, pos_emb,
            cs_input,  # The (possibly masked) CS values
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
    log_interval = max(1, len(loader) // 4)
    
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
        residues = batch['residues'].to(device)
        ss_types = batch['ss_types'].to(device)
        cs_input = batch['cs_input'].to(device)
        mystery = batch['mystery'].to(device)
        phi = batch['phi'].to(device)
        psi = batch['psi'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        cs_target = batch['cs_target'].to(device)
        cs_mask = batch['cs_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - model sees cs_input (masked values)
        predictions = model(residues, ss_types, cs_input, mystery, phi, psi, attn_mask)
        
        # Compute loss ONLY on masked positions
        if cs_mask.sum() == 0:
            continue  # Skip if no masked values
            
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
        
        samples_per_sec = len(batch['residues']) / batch_time if batch_time > 0 else 0
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'grad': f'{grad_norm:.3f}',
            'masked': f'{batch_count}',
            'samples/s': f'{samples_per_sec:.0f}'
        })
        
        # Mid-epoch logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_count if total_count > 0 else 0
            avg_time = np.mean(batch_times[-log_interval:])
            avg_throughput = BATCH_SIZE / avg_time if avg_time > 0 else 0
            print(f"\n  Batch {batch_idx+1}/{len(loader)} - Loss: {avg_loss:.4f} | "
                  f"GradNorm: {grad_norm:.4f} | Masked: {batch_count} | {avg_throughput:.0f} samples/s")
    
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
            residues = batch['residues'].to(device)
            ss_types = batch['ss_types'].to(device)
            cs_input = batch['cs_input'].to(device)
            mystery = batch['mystery'].to(device)
            phi = batch['phi'].to(device)
            psi = batch['psi'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            cs_target = batch['cs_target'].to(device)
            cs_mask = batch['cs_mask'].to(device)
            
            predictions = model(residues, ss_types, cs_input, mystery, phi, psi, attn_mask)
            
            if cs_mask.sum() == 0:
                continue
                
            loss = F.mse_loss(predictions[cs_mask], cs_target[cs_mask])
            
            batch_count = cs_mask.sum().item()
            total_loss += loss.item() * batch_count
            total_count += batch_count
            
            # Per-CS type errors (in original scale)
            for i, col in enumerate(cs_cols):
                mask_i = cs_mask[:, :, i]
                if mask_i.sum() > 0:
                    pred_i = predictions[:, :, i][mask_i]
                    true_i = cs_target[:, :, i][mask_i]
                    
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
    
    # Load and split data
    train_df, test_df = load_and_split_data(DATA_PATH)
    
    # Compute statistics
    stats = compute_stats(train_df)
    
    # Create datasets
    train_dataset = ProteinDataset(train_df, stats, max_seq_len=MAX_SEQ_LEN, 
                                   mask_prob=MASK_PROB, training=True, verbose=True)
    test_dataset = ProteinDataset(test_df, stats, max_seq_len=MAX_SEQ_LEN, 
                                  mask_prob=MASK_PROB, training=True, verbose=False)  # Same masking at test
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=16,
        pin_memory=True,
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
    print("Starting training with BERT-style masking...")
    print(f"Masking {MASK_PROB*100:.0f}% of known values per epoch")
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