#!/usr/bin/env python3
"""
OPTIMIZED Chemical Shift Prediction Training
Fixes:
1. Multiprocessing start method set to 'spawn'
2. Optimal worker count (8 instead of 20)
3. Pre-processing moved out of __getitem__
4. Mixed precision training
5. Larger batch size for RTX 5090
6. Non-blocking CUDA transfers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
from tqdm import tqdm
import os
import math
import wandb
from collections import defaultdict
import torch.multiprocessing as mp
import gc

# CRITICAL FIX #1: Set multiprocessing start method BEFORE any other imports
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("✓ Multiprocessing set to 'spawn' mode")
    except RuntimeError:
        print("⚠ Multiprocessing method already set")

# Configuration
DATA_DIR = 'preprocessed_data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.pkl')
EVAL_FILE = os.path.join(DATA_DIR, 'eval_data.pkl')
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.pkl')

# OPTIMIZED: Larger batch size for RTX 5090
BATCH_SIZE = 512  # Up from 16
EPOCHS = 200
LEARNING_RATE = 2.5e-5  # Slightly increased for larger batch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MASK_PROB = 0.15

# Model hyperparameters
EMBED_DIM = 64
HIDDEN_DIM = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1
MAX_DIST = 20.0

# Wandb
WANDB_PROJECT = "chemical-shift-prediction"
WANDB_RUN_NAME = "atom-transformer-v1-optimized"

# OPTIMIZED: Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"✓ CUDA optimizations enabled")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Global data - loaded once in main process
_TRAIN_DATA = None
_EVAL_DATA = None
_METADATA = None

# Shared-memory numeric storage
_TRAIN_SHARED = None
_EVAL_SHARED = None


def _build_shared_from_list(data, metadata):
    """
    Take the list-of-dicts dataset and pack all numeric fields into large
    shared-memory tensors so multiple DataLoader workers see ONE physical copy.
    """
    print("Building shared-memory tensors...")
    stats = metadata['normalization_stats']
    backbone_atoms = metadata['backbone_atoms']

    n_samples = len(data)
    max_atoms = max(s['n_atoms'] for s in data)

    # Pack related features into a few large shared tensors to reduce the number
    # of shared-memory segments (and thus open file descriptors).
    # int_features: [atom_type, atom_name, residue_type, ss_type, relative_position]
    int_features = torch.empty((n_samples, max_atoms, 5), dtype=torch.long).share_memory_()
    # float_features: [coord_x, coord_y, coord_z, phi, psi, cs_normalized, cs_err_normalized]
    float_features = torch.empty((n_samples, max_atoms, 7), dtype=torch.float32).share_memory_()
    # bool_features: [cs_mask, coords_mask]
    bool_features = torch.empty((n_samples, max_atoms, 2), dtype=torch.bool).share_memory_()
    # distance matrix stays as its own tensor
    distance_matrix = torch.empty((n_samples, max_atoms, max_atoms), dtype=torch.float32).share_memory_()
    n_atoms_tensor = torch.empty((n_samples,), dtype=torch.int32).share_memory_()

    # Small per-sample metadata stored as Python lists (no need for shared tensors)
    atom_name_str_all = []
    center_indices_all = []

    for i, sample in enumerate(tqdm(data, desc="Packing into shared tensors")):
        n_atoms = sample['n_atoms']
        n_atoms_tensor[i] = n_atoms

        # Create views for this sample from packed tensors
        int_feat = int_features[i, :n_atoms]      # shape [n_atoms, 5]
        float_feat = float_features[i, :n_atoms]  # shape [n_atoms, 7]
        bool_feat = bool_features[i, :n_atoms]    # shape [n_atoms, 2]

        dm = distance_matrix[i, :n_atoms, :n_atoms]

        # Copy categorical / indices into int_features
        at = torch.as_tensor(sample['atom_type'], dtype=torch.long)
        an = torch.as_tensor(sample['atom_name'], dtype=torch.long)
        rt = torch.as_tensor(sample['residue_type'], dtype=torch.long)
        ss = torch.as_tensor(sample['ss_type'], dtype=torch.long)
        rp = torch.as_tensor(sample['relative_position'], dtype=torch.long)
        int_feat[:, 0].copy_(at)
        int_feat[:, 1].copy_(an)
        int_feat[:, 2].copy_(rt)
        int_feat[:, 3].copy_(ss)
        int_feat[:, 4].copy_(rp)

        # Normalize coordinates
        coords_np = sample['coords'].astype(np.float32, copy=True)
        coords_m = sample['coords_mask']
        coords_np[coords_m] = (coords_np[coords_m] - stats['coords_mean']) / (stats['coords_std'] + 1e-8)

        # Normalize angles
        phi_np = ((sample['phi'] - stats['phi_mean']) / (stats['phi_std'] + 1e-8)).astype(np.float32)
        psi_np = ((sample['psi'] - stats['psi_mean']) / (stats['psi_std'] + 1e-8)).astype(np.float32)

        # Write float features: [coord_x, coord_y, coord_z, phi, psi, cs_normalized, cs_err_normalized]
        float_feat[:, 0].copy_(torch.from_numpy(coords_np[:, 0]))
        float_feat[:, 1].copy_(torch.from_numpy(coords_np[:, 1]))
        float_feat[:, 2].copy_(torch.from_numpy(coords_np[:, 2]))
        float_feat[:, 3].copy_(torch.from_numpy(phi_np))
        float_feat[:, 4].copy_(torch.from_numpy(psi_np))

        # CS normalization
        cs_vals = sample['cs_values'].astype(np.float32)
        cs_errs = sample['cs_errors'].astype(np.float32)
        cs_m_np = sample['cs_mask'].astype(bool)
        atom_names = np.asarray(sample['atom_name_str'])

        cs_n_np = cs_vals.copy()
        cs_e_np = cs_errs.copy()

        for atom_name_str in backbone_atoms:
            mean_key = f"{atom_name_str}_mean"
            std_key = f"{atom_name_str}_std"
            if mean_key not in stats or std_key not in stats:
                continue

            mask = (atom_names == atom_name_str) & cs_m_np
            if not np.any(mask):
                continue

            mean = stats[mean_key]
            std = stats[std_key] + 1e-8
            cs_n_np[mask] = (cs_vals[mask] - mean) / std
            cs_e_np[mask] = cs_errs[mask] / std

        # Write CS and masks into packed tensors
        float_feat[:, 5].copy_(torch.from_numpy(cs_n_np))
        float_feat[:, 6].copy_(torch.from_numpy(cs_e_np))
        bool_feat[:, 0].copy_(torch.from_numpy(cs_m_np.astype(bool)))
        bool_feat[:, 1].copy_(torch.from_numpy(coords_m.astype(bool)))

        # Normalize distance matrix
        dist = sample['distance_matrix'].astype(np.float32, copy=True)
        dist = np.clip(dist, 0, MAX_DIST)
        dist = dist / (stats['dist_std'] + 1e-8)
        dist = np.clip(dist, 0, 10)
        dm.copy_(torch.from_numpy(dist))

        # center_indices (remains as Python list)
        center_indices_all.append(sample['center_indices'])

        atom_name_str_all.append(sample['atom_name_str'])

        # Zero-pad tail if this sample has fewer than max_atoms
        if n_atoms < max_atoms:
            int_features[i, n_atoms:].zero_()
            float_features[i, n_atoms:].zero_()
            bool_features[i, n_atoms:] = False
            distance_matrix[i, n_atoms:, :].zero_()
            distance_matrix[i, :, n_atoms:].zero_()

    return {
        'int_features': int_features,
        'float_features': float_features,
        'bool_features': bool_features,
        'distance_matrix': distance_matrix,
        'center_indices': center_indices_all,
        'n_atoms': n_atoms_tensor,
        'atom_name_str': atom_name_str_all,
    }


def load_global_data():
    """Load metadata and build shared-memory tensors for train/eval once."""
    global _TRAIN_DATA, _EVAL_DATA, _METADATA, _TRAIN_SHARED, _EVAL_SHARED

    if _METADATA is None:
        print("Loading metadata...")
        with open(METADATA_FILE, 'rb') as f:
            _METADATA = pickle.load(f)

    # Build shared train tensors
    if _TRAIN_SHARED is None:
        if _TRAIN_DATA is None:
            print("Loading train data...")
            with open(TRAIN_FILE, 'rb') as f:
                _TRAIN_DATA = pickle.load(f)
        _TRAIN_SHARED = _build_shared_from_list(_TRAIN_DATA, _METADATA)
        _TRAIN_DATA = None
        gc.collect()

    # Build shared eval tensors
    if _EVAL_SHARED is None:
        if _EVAL_DATA is None:
            print("Loading eval data...")
            with open(EVAL_FILE, 'rb') as f:
                _EVAL_DATA = pickle.load(f)
        _EVAL_SHARED = _build_shared_from_list(_EVAL_DATA, _METADATA)
        _EVAL_DATA = None
        gc.collect()

# ============================================================================
# Model Components (unchanged)
# ============================================================================

class GaussianDistanceEncoding(nn.Module):
    def __init__(self, n_kernels=64, max_dist=20.0):
        super().__init__()
        self.max_dist = max_dist
        self.n_kernels = n_kernels
        
        centers = torch.linspace(0, max_dist, n_kernels)
        self.register_buffer('centers', centers)
        self.widths = nn.Parameter(torch.ones(n_kernels) * 0.5)
        
    def forward(self, distances):
        distances = torch.clamp(distances, 0, self.max_dist)
        distances = distances.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        widths = torch.clamp(torch.abs(self.widths), min=0.1, max=5.0).view(1, 1, 1, -1)
        encoded = torch.exp(-((distances - centers) ** 2) / (2 * widths ** 2))
        encoded = torch.clamp(encoded, min=1e-8, max=1.0)
        return encoded

class DistanceAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dist_proj = nn.Linear(64, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, dist_encoding, mask=None):
        batch_size, n_atoms, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, n_atoms, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_atoms, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_atoms, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        dist_bias = self.dist_proj(dist_encoding).permute(0, 3, 1, 2)
        scores = scores + dist_bias
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, -1e9)
            query_mask = mask.unsqueeze(1).unsqueeze(-1)
        
        attn = F.softmax(scores, dim=-1)
        
        if mask is not None:
            attn = attn * query_mask.float()
        
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_atoms, self.hidden_dim)
        out = self.o_proj(out)
        
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = DistanceAwareMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, dist_encoding, mask=None):
        x = x + self.attn(self.norm1(x), dist_encoding, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class AtomTransformerCS(nn.Module):
    def __init__(self, metadata, embed_dim=64, hidden_dim=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        
        encoders = metadata['encoders']
        self.n_atom_types = len(encoders['atom_type'].classes_)
        self.n_atom_names = len(encoders['atom_name'].classes_)
        self.n_residues = len(encoders['residue'].classes_)
        self.n_ss = len(encoders['ss'].classes_)
        self.n_positions = 2 * metadata['context_window'] + 1
        
        self.atom_type_embed = nn.Embedding(self.n_atom_types, embed_dim)
        self.atom_name_embed = nn.Embedding(self.n_atom_names, embed_dim)
        self.residue_embed = nn.Embedding(self.n_residues, embed_dim)
        self.ss_embed = nn.Embedding(self.n_ss, embed_dim)
        self.position_embed = nn.Embedding(self.n_positions + 10, embed_dim)
        
        self.distance_encoder = GaussianDistanceEncoding(n_kernels=64, max_dist=MAX_DIST)
        self.continuous_proj = nn.Linear(6, embed_dim)
        
        self.input_proj = nn.Sequential(
            nn.Linear(6 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.backbone_atoms = metadata['backbone_atoms']
        self.cs_heads = nn.ModuleDict()
        
        for atom_name in self.backbone_atoms:
            self.cs_heads[atom_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        self.atom_name_to_idx = {name: i for i, name in enumerate(encoders['atom_name'].classes_)}
        self.idx_to_atom_name = {i: name for name, i in self.atom_name_to_idx.items()}
        
    def forward(self, batch):
        atom_type_emb = self.atom_type_embed(batch['atom_type'])
        atom_name_emb = self.atom_name_embed(batch['atom_name'])
        residue_emb = self.residue_embed(batch['residue_type'])
        ss_emb = self.ss_embed(batch['ss_type'])
        
        pos_indices = batch['relative_position'] + (self.n_positions // 2)
        pos_emb = self.position_embed(pos_indices)
        
        continuous = torch.stack([
            batch['coords'][:, :, 0],
            batch['coords'][:, :, 1],
            batch['coords'][:, :, 2],
            batch['phi'],
            batch['psi'],
            batch['cs_input']
        ], dim=-1)
        continuous_emb = self.continuous_proj(continuous)
        
        x = torch.cat([
            atom_type_emb, atom_name_emb, residue_emb, 
            ss_emb, pos_emb, continuous_emb
        ], dim=-1)
        x = self.input_proj(x)
        
        dist_encoding = self.distance_encoder(batch['distance_matrix'])
        
        for layer in self.layers:
            x = layer(x, dist_encoding, batch['atom_mask'])
        
        batch_size, max_atoms = batch['atom_name'].shape
        predictions = torch.zeros(batch_size, max_atoms, device=x.device, dtype=x.dtype)
        
        for atom_name in self.backbone_atoms:
            if atom_name not in self.atom_name_to_idx:
                continue
                
            atom_idx = self.atom_name_to_idx[atom_name]
            mask = (batch['atom_name'] == atom_idx)
            
            if mask.any():
                atom_features = x[mask]
                atom_preds = self.cs_heads[atom_name](atom_features).squeeze(-1)
                predictions[mask] = atom_preds
        
        return predictions

# ============================================================================
# OPTIMIZED Dataset - Pre-compute all normalization
# ============================================================================


class AtomDataset(Dataset):
    """
    Dataset that reads from shared-memory tensors built in load_global_data().
    This gives one physical copy of the numeric data while allowing many
    DataLoader workers to slice from it in parallel.
    """
    def __init__(self, is_train=True, mask_prob=0.15, training=True, seed=None):
        load_global_data()

        self.is_train = is_train
        self.mask_prob = mask_prob
        self.training = training
        self.seed = seed
        self.stats = _METADATA['normalization_stats']
        self.backbone_atoms = _METADATA['backbone_atoms']

        self.shared = _TRAIN_SHARED if is_train else _EVAL_SHARED
        self.n_samples = self.shared['n_atoms'].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sh = self.shared
        n_atoms = int(sh['n_atoms'][idx])

        int_feat = sh['int_features'][idx, :n_atoms]      # [n_atoms, 5]
        float_feat = sh['float_features'][idx, :n_atoms]  # [n_atoms, 7]
        bool_feat = sh['bool_features'][idx, :n_atoms]    # [n_atoms, 2]

        atom_type = int_feat[:, 0]
        atom_name = int_feat[:, 1]
        residue_type = int_feat[:, 2]
        ss_type = int_feat[:, 3]
        relative_position = int_feat[:, 4]

        coords = float_feat[:, 0:3]
        phi = float_feat[:, 3]
        psi = float_feat[:, 4]
        cs_norm = float_feat[:, 5]
        cs_err_norm = float_feat[:, 6]

        cs_mask = bool_feat[:, 0]
        coords_mask = bool_feat[:, 1]

        dist = sh['distance_matrix'][idx, :n_atoms, :n_atoms]
        center_idx = sh['center_indices'][idx]

        # BERT-style masking (done per-sample)
        cs_input = cs_norm.clone()
        cs_target = cs_norm.clone()
        prediction_mask = torch.zeros(n_atoms, dtype=torch.bool)

        if self.training:
            mask_probs = torch.rand(n_atoms)
        else:
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + idx)
                mask_probs = torch.from_numpy(rng.random(n_atoms).astype(np.float32))
            else:
                mask_probs = torch.rand(n_atoms)

        mask_these = cs_mask & (mask_probs < self.mask_prob)
        cs_input[mask_these] = 0.0
        prediction_mask[mask_these] = True

        return {
            'atom_type': atom_type,
            'atom_name': atom_name,
            'atom_name_str': self.shared['atom_name_str'][idx],
            'residue_type': residue_type,
            'ss_type': ss_type,
            'relative_position': relative_position,
            'coords': coords,
            'phi': phi,
            'psi': psi,
            'cs_input': cs_input,
            'cs_target': cs_target,
            'cs_errors': cs_err_norm,
            'distance_matrix': dist,
            'atom_mask': coords_mask,
            'prediction_mask': prediction_mask,
            'center_indices': center_idx,
            'n_atoms': n_atoms,
        }

def collate_fn(batch):
    max_atoms = max(b['n_atoms'] for b in batch)
    batch_size = len(batch)
    
    collated = {
        'atom_type': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'atom_name': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'residue_type': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'ss_type': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'relative_position': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'coords': torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32),
        'phi': torch.zeros(batch_size, max_atoms, dtype=torch.float32),
        'psi': torch.zeros(batch_size, max_atoms, dtype=torch.float32),
        'cs_input': torch.zeros(batch_size, max_atoms, dtype=torch.float32),
        'cs_target': torch.zeros(batch_size, max_atoms, dtype=torch.float32),
        'cs_errors': torch.zeros(batch_size, max_atoms, dtype=torch.float32),
        'distance_matrix': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float32),
        'atom_mask': torch.zeros(batch_size, max_atoms, dtype=torch.bool),
        'prediction_mask': torch.zeros(batch_size, max_atoms, dtype=torch.bool),
    }
    
    atom_name_strs = []
    
    for i, b in enumerate(batch):
        n = b['n_atoms']
        atom_name_strs.append(b['atom_name_str'])
        for key in collated.keys():
            if key == 'distance_matrix':
                collated[key][i, :n, :n] = b[key]
            elif key in b:
                collated[key][i, :n] = b[key]
    
    collated['atom_name_str'] = atom_name_strs
    return collated

# ============================================================================
# Loss
# ============================================================================

def uncertainty_weighted_loss(pred, target, uncertainty, pred_mask, atom_name_strs, backbone_atoms):
    batch_size, max_atoms = pred.shape
    backbone_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
    
    for i in range(batch_size):
        for j in range(max_atoms):
            if j < len(atom_name_strs[i]):
                if atom_name_strs[i][j] in backbone_atoms:
                    backbone_mask[i, j] = True
    
    final_mask = pred_mask & backbone_mask
    
    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    uncertainty = torch.clamp(uncertainty, min=0.01, max=10.0)
    weights = 1.0 / (uncertainty[final_mask] ** 2 + 1e-3)
    weights = torch.clamp(weights, min=0.01, max=100.0)
    weights = weights / (weights.mean() + 1e-8)
    
    errors = (pred[final_mask] - target[final_mask]) ** 2
    loss = (errors * weights).mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device)
    
    return loss

# ============================================================================
# Training with Mixed Precision
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, device, epoch, stats, backbone_atoms):
    model.train()
    total_loss = 0
    total_count = 0
    atom_errors = defaultdict(list)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    log_interval = max(1, len(loader) // 4)
    
    for batch_idx, batch in enumerate(pbar):
        atom_name_strs = batch.pop('atom_name_str')
        
        # OPTIMIZED: non_blocking transfer
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            predictions = model(batch)
            
            if torch.isnan(predictions).any():
                print(f"⚠ NaN in predictions at batch {batch_idx}, skipping")
                continue
            
            loss = uncertainty_weighted_loss(
                predictions, batch['cs_target'], batch['cs_errors'],
                batch['prediction_mask'], atom_name_strs, backbone_atoms
            )
        
        if loss.item() > 0 and not torch.isnan(loss):
            scaler.scale(loss).backward()
            
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            batch_size, max_atoms = predictions.shape
            backbone_mask = torch.zeros_like(batch['prediction_mask'], dtype=torch.bool)
            for i in range(batch_size):
                for j in range(max_atoms):
                    if j < len(atom_name_strs[i]):
                        if atom_name_strs[i][j] in backbone_atoms:
                            backbone_mask[i, j] = True
            
            final_mask = batch['prediction_mask'] & backbone_mask
            batch_count = final_mask.sum().item()
            
            total_loss += loss.item() * batch_count
            total_count += batch_count
            
            with torch.no_grad():
                for i in range(batch_size):
                    for j in range(max_atoms):
                        if final_mask[i, j] and j < len(atom_name_strs[i]):
                            atom_name = atom_name_strs[i][j]
                            if atom_name in backbone_atoms:
                                mean_key = f'{atom_name}_mean'
                                std_key = f'{atom_name}_std'
                                
                                pred_val = predictions[i, j].item() * stats[std_key] + stats[mean_key]
                                target_val = batch['cs_target'][i, j].item() * stats[std_key] + stats[mean_key]
                                error = abs(pred_val - target_val)
                                atom_errors[atom_name].append(error)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.3f}'})
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / total_count if total_count > 0 else 0
                overall_mae = np.mean([e for errors in atom_errors.values() for e in errors]) if atom_errors else 0
                
                print(f"\n  Progress: {(batch_idx+1)/len(loader)*100:.0f}% | Loss: {avg_loss:.4f} | MAE: {overall_mae:.3f} ppm")
                print("  Per-atom MAE:")
                for atom_name in backbone_atoms:
                    if atom_name in atom_errors and len(atom_errors[atom_name]) > 0:
                        mae = np.mean(atom_errors[atom_name])
                        print(f"    {atom_name:8s}: {mae:.3f} ppm")
                print()
    
    return total_loss / total_count if total_count > 0 else 0

def evaluate(model, loader, device, stats, backbone_atoms):
    model.eval()
    total_loss = 0
    total_count = 0
    atom_errors = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            atom_name_strs = batch.pop('atom_name_str')
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            with autocast():
                predictions = model(batch)
                loss = uncertainty_weighted_loss(
                    predictions, batch['cs_target'], batch['cs_errors'],
                    batch['prediction_mask'], atom_name_strs, backbone_atoms
                )
            
            batch_size, max_atoms = predictions.shape
            backbone_mask = torch.zeros_like(batch['prediction_mask'], dtype=torch.bool)
            for i in range(batch_size):
                for j in range(max_atoms):
                    if j < len(atom_name_strs[i]):
                        if atom_name_strs[i][j] in backbone_atoms:
                            backbone_mask[i, j] = True
            
            final_mask = batch['prediction_mask'] & backbone_mask
            batch_count = final_mask.sum().item()
            
            if batch_count > 0:
                total_loss += loss.item() * batch_count
                total_count += batch_count
                
                for i in range(batch_size):
                    for j in range(max_atoms):
                        if final_mask[i, j] and j < len(atom_name_strs[i]):
                            atom_name = atom_name_strs[i][j]
                            if atom_name in backbone_atoms:
                                mean_key = f'{atom_name}_mean'
                                std_key = f'{atom_name}_std'
                                
                                pred_val = predictions[i, j].item() * stats[std_key] + stats[mean_key]
                                target_val = batch['cs_target'][i, j].item() * stats[std_key] + stats[mean_key]
                                error = abs(pred_val - target_val)
                                atom_errors[atom_name].append(error)
    
    avg_loss = total_loss / total_count if total_count > 0 else 0
    
    per_atom_mae = {}
    print("\n" + "="*80)
    print("PER-ATOM MAE (ppm):")
    print("="*80)
    for atom_name in backbone_atoms:
        if atom_name in atom_errors and len(atom_errors[atom_name]) > 0:
            mae = np.mean(atom_errors[atom_name])
            per_atom_mae[f'mae/{atom_name}'] = mae
            print(f"  {atom_name:8s}: {mae:.3f} ppm ({len(atom_errors[atom_name])} predictions)")
        else:
            print(f"  {atom_name:8s}: No predictions")
    
    overall_mae = np.mean([e for errors in atom_errors.values() for e in errors]) if atom_errors else 0
    print(f"\n  Overall: {overall_mae:.3f} ppm")
    print("="*80)
    
    return avg_loss, per_atom_mae, overall_mae

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("ATOM TRANSFORMER FOR CHEMICAL SHIFT PREDICTION (OPTIMIZED)")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Batch Size: {BATCH_SIZE}")
    
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "mask_prob": MASK_PROB,
            "max_dist": MAX_DIST,
            "mixed_precision": True,
            "optimizations": "spawn+preprocess+amp+larger_batch"
        }
    )
    
    # Load data in main process
    load_global_data()
    
    print(f"Train samples: {_METADATA['n_train_samples']:,}")
    print(f"Eval samples: {_METADATA['n_eval_samples']:,}")
    print(f"Backbone atoms: {', '.join(_METADATA['backbone_atoms'])}")
    
    # Datasets backed by shared-memory tensors
    train_dataset = AtomDataset(is_train=True, mask_prob=MASK_PROB, training=True)
    eval_dataset = AtomDataset(is_train=False, mask_prob=MASK_PROB, training=False, seed=42)
    
    # CRITICAL FIX #2: Optimal worker count (8 is usually best, not 20)
    # Too many workers causes contention and actually slows things down
    num_workers = 4
    print(f"\n🚀 Using {num_workers} dataloader workers")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")
    
    model = AtomTransformerCS(
        _METADATA, 
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    wandb.config.update({"model_parameters": total_params})
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-"*80)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, DEVICE, epoch,
            _METADATA['normalization_stats'], _METADATA['backbone_atoms']
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        eval_loss, per_atom_mae, overall_mae = evaluate(
            model, eval_loader, DEVICE, 
            _METADATA['normalization_stats'],
            _METADATA['backbone_atoms']
        )
        
        scheduler.step(eval_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        log_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_mae_overall': overall_mae,
            'learning_rate': current_lr
        }
        log_dict.update(per_atom_mae)
        wandb.log(log_dict)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_atom_transformer.pt')
            wandb.save('best_atom_transformer.pt')
            print(f"✓ New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{early_stop_patience})")
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("\n" + "="*80)
    print(f"Training complete! Best eval loss: {best_loss:.4f}")
    print("="*80)
    
    wandb.finish()

if __name__ == '__main__':
    main()
