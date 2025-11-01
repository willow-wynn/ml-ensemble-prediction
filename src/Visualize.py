import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
DATA_PATH = '/Users/wynndiaz/he_lab/data/improved.csv'
MODEL_PATH = 'best_model_cnn.pt'
BATCH_SIZE = 256
MAX_SEQ_LEN = 256
CONTEXT_WINDOW = 9
MASK_PROB = 0.2
DEVICE = 'cpu'  # Use CPU for visualization to avoid CUDA issues

# ============================================================================
# Model Architecture (same as training)
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
        
        layers = []
        in_ch = in_feat
        for out_ch in channels:
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        self.attn = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
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
        
    def forward(self, x, return_attention=False):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        attn_scores = self.attn(x)
        attn_w = F.softmax(attn_scores, dim=1)
        x_pooled = torch.sum(x * attn_w, dim=1)
        
        predictions = []
        for head in self.cs_heads:
            predictions.append(head(x_pooled).squeeze(-1))
        
        predictions = torch.stack(predictions, dim=-1)
        
        if return_attention:
            return predictions, attn_w
        return predictions

# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_data(csv_path, test_size=0.15, random_state=42):
    """Load data and split by protein ID"""
    print("Loading data...")
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows")
    
    df.columns = ['protein_id', 'position', 'residue', 'CS1', 'CS2', 
                  'CS3', 'CS4', 'ss_type', 'mystery_feature', 'phi', 'psi']
    
    df['protein_id'] = pd.to_numeric(df['protein_id'], errors='coerce').astype('Int64')
    df['position'] = pd.to_numeric(df['position'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['protein_id', 'position'])
    
    for col in ['CS1', 'CS2', 'CS3', 'CS4', 'mystery_feature', 'phi', 'psi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    protein_ids = df['protein_id'].unique()
    train_proteins, test_proteins = train_test_split(
        protein_ids, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df['protein_id'].isin(train_proteins)].copy()
    test_df = df[df['protein_id'].isin(test_proteins)].copy()
    
    return train_df, test_df

def compute_stats(df):
    """Compute normalization statistics"""
    stats = {}
    cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
    
    for col in cs_cols:
        valid = df[df[col] != 9999.0][col]
        stats[col] = {'mean': float(valid.mean()), 'std': float(valid.std())}
    
    stats['mystery_feature'] = {
        'mean': float(df['mystery_feature'].mean()),
        'std': float(df['mystery_feature'].std())
    }
    
    for col in ['phi', 'psi']:
        valid = df[df[col] != 9999.0][col]
        stats[col] = {'mean': float(valid.mean()), 'std': float(valid.std())}
    
    return stats

# ============================================================================
# Dataset
# ============================================================================

class ProteinDataset(Dataset):
    def __init__(self, df, stats, context_window=3, mask_prob=0.2, training=True, 
                 residue_encoder=None, ss_encoder=None, seed=None):
        self.df = df
        self.stats = stats
        self.context_window = context_window
        self.mask_prob = mask_prob
        self.training = training
        self.seed = seed
        self.cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
        
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
        
        self.proteins = list(df.groupby('protein_id'))
        
        self.samples = []
        for protein_id, prot_df in self.proteins:
            prot_df = prot_df.sort_values('position').reset_index(drop=True)
            for i in range(context_window, len(prot_df) - context_window):
                window_start = i - context_window
                window_end = i + context_window
                
                start_pos = prot_df.iloc[window_start]['position']
                end_pos = prot_df.iloc[window_end]['position']
                expected_span = 2 * context_window
                actual_span = end_pos - start_pos
                
                if actual_span == expected_span:
                    self.samples.append((protein_id, i))
        
    def normalize(self, value, col):
        if value == 9999.0 or np.isnan(value):
            return 0.0
        return (value - self.stats[col]['mean']) / (self.stats[col]['std'] + 1e-8)
    
    def denormalize(self, value, col):
        return value * self.stats[col]['std'] + self.stats[col]['mean']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        protein_id, center_idx = self.samples[idx]
        prot_df = self.df[self.df['protein_id'] == protein_id].sort_values('position').reset_index(drop=True)
        
        start_idx = center_idx - self.context_window
        end_idx = center_idx + self.context_window + 1
        window_df = prot_df.iloc[start_idx:end_idx]
        
        seq_len = len(window_df)
        
        # Encode residues
        residues_onehot = torch.zeros(seq_len, self.n_residue_types, dtype=torch.float32)
        for i, res in enumerate(window_df['residue']):
            if res in self.residue_encoder.classes_:
                residues_onehot[i, self.residue_encoder.transform([res])[0]] = 1.0
        
        # Encode SS
        ss_onehot = torch.zeros(seq_len, self.n_ss_types, dtype=torch.float32)
        for i, ss in enumerate(window_df['ss_type']):
            if ss in self.ss_encoder.classes_:
                ss_onehot[i, self.ss_encoder.transform([ss])[0]] = 1.0
        
        # Chemical shifts
        cs_input = torch.zeros(seq_len, 4, dtype=torch.float32)
        cs_target = torch.zeros(seq_len, 4, dtype=torch.float32)
        cs_mask = torch.zeros(seq_len, 4, dtype=torch.bool)
        
        for i, col in enumerate(self.cs_cols):
            values = window_df[col].values
            available = (values != 9999.0)
            
            normalized = torch.tensor([self.normalize(v, col) for v in values], dtype=torch.float32)
            cs_target[:, i] = normalized
            
            if self.training:
                mask_these = available & (np.random.random(seq_len) < self.mask_prob)
                cs_mask[:, i] = torch.tensor(mask_these)
                mask_these_tensor = torch.tensor(mask_these, dtype=torch.bool)
                cs_input[:, i] = normalized * (~mask_these_tensor).float()
            else:
                if self.seed is not None:
                    rng = np.random.RandomState(self.seed + idx)
                    mask_these = available & (rng.random(seq_len) < self.mask_prob)
                else:
                    mask_these = available & (np.random.random(seq_len) < self.mask_prob)
                
                cs_mask[:, i] = torch.tensor(mask_these)
                mask_these_tensor = torch.tensor(mask_these, dtype=torch.bool)
                cs_input[:, i] = normalized * (~mask_these_tensor).float()
        
        # Other features
        mystery = torch.tensor([
            self.normalize(v, 'mystery_feature') for v in window_df['mystery_feature'].values
        ], dtype=torch.float32)
        
        phi = torch.tensor([self.normalize(v, 'phi') for v in window_df['phi'].values], dtype=torch.float32)
        psi = torch.tensor([self.normalize(v, 'psi') for v in window_df['psi'].values], dtype=torch.float32)
        
        features = torch.cat([
            residues_onehot, ss_onehot, cs_input,
            mystery.unsqueeze(-1), phi.unsqueeze(-1), psi.unsqueeze(-1)
        ], dim=-1)
        
        center_in_window = self.context_window
        cs_target_center = cs_target[center_in_window]
        cs_mask_center = cs_mask[center_in_window]
        
        # Store metadata for visualization
        center_residue = window_df.iloc[center_in_window]['residue']
        center_ss = window_df.iloc[center_in_window]['ss_type']
        
        return {
            'features': features,
            'cs_target': cs_target_center,
            'cs_mask': cs_mask_center,
            'residue': center_residue,
            'ss_type': center_ss,
            'protein_id': protein_id
        }

# ============================================================================
# Comprehensive Analysis Functions
# ============================================================================

def analyze_predictions(model, loader, device, stats, cs_cols, save_dir='analysis_viz'):
    """Comprehensive prediction analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    results = {
        'predictions': [],
        'targets': [],
        'errors': [],
        'residues': [],
        'ss_types': [],
        'masks': [],
        'attention_weights': [],
        'protein_ids': []
    }
    
    print("\nCollecting predictions and attention patterns...")
    with torch.no_grad():
        for batch in tqdm(loader):
            features = batch['features'].to(device)
            cs_target = batch['cs_target'].to(device)
            cs_mask = batch['cs_mask'].to(device)
            
            predictions, attention = model(features, return_attention=True)
            
            results['predictions'].append(predictions.cpu().numpy())
            results['targets'].append(cs_target.cpu().numpy())
            results['masks'].append(cs_mask.cpu().numpy())
            results['residues'].extend(batch['residue'])
            results['ss_types'].extend(batch['ss_type'])
            results['attention_weights'].append(attention.cpu().numpy())
            results['protein_ids'].extend([int(pid) for pid in batch['protein_id']])
    
    # Concatenate results
    predictions = np.concatenate(results['predictions'])
    targets = np.concatenate(results['targets'])
    masks = np.concatenate(results['masks'])
    attention_weights = np.concatenate(results['attention_weights'])
    
    # Denormalize
    predictions_denorm = np.zeros_like(predictions)
    targets_denorm = np.zeros_like(targets)
    
    for i, col in enumerate(cs_cols):
        predictions_denorm[:, i] = predictions[:, i] * stats[col]['std'] + stats[col]['mean']
        targets_denorm[:, i] = targets[:, i] * stats[col]['std'] + stats[col]['mean']
    
    errors = np.abs(predictions_denorm - targets_denorm)
    
    return {
        'predictions': predictions_denorm,
        'targets': targets_denorm,
        'errors': errors,
        'masks': masks,
        'residues': np.array(results['residues']),
        'ss_types': np.array(results['ss_types']),
        'attention': attention_weights,
        'protein_ids': np.array(results['protein_ids'])
    }

def plot_prediction_scatter(results, cs_cols, save_dir):
    """Scatter plots of predictions vs targets"""
    print("\nCreating prediction scatter plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
        ax = axes[i]
        
        mask = results['masks'][:, i]
        pred = results['predictions'][mask, i]
        target = results['targets'][mask, i]
        
        # Density scatter plot
        from scipy.stats import gaussian_kde
        
        if len(pred) > 1000:
            # Sample for density calculation
            sample_idx = np.random.choice(len(pred), 1000, replace=False)
            xy = np.vstack([target[sample_idx], pred[sample_idx]])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_plot, y_plot, z_plot = target[sample_idx][idx], pred[sample_idx][idx], z[idx]
        else:
            x_plot, y_plot = target, pred
            z_plot = None
        
        if z_plot is not None:
            scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=20, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Density')
        else:
            ax.scatter(target, pred, alpha=0.3, s=20)
        
        # Perfect prediction line
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Statistics
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target)**2))
        r2 = np.corrcoef(pred, target)[0, 1]**2
        
        ax.set_xlabel(f'True {name} (ppm)', fontsize=12)
        ax.set_ylabel(f'Predicted {name} (ppm)', fontsize=12)
        ax.set_title(f'{name}\nMAE={mae:.3f} ppm, RMSE={rmse:.3f} ppm, R²={r2:.3f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved prediction_scatter.png")

def plot_error_distributions(results, cs_cols, save_dir):
    """Error distribution analysis"""
    print("\nCreating error distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
        ax = axes[i]
        
        mask = results['masks'][:, i]
        errors = results['errors'][mask, i]
        
        # Histogram
        ax.hist(errors, bins=100, alpha=0.7, edgecolor='black', density=True)
        
        # Statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        p95_error = np.percentile(errors, 95)
        
        ax.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
        ax.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f}')
        ax.axvline(p95_error, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95_error:.3f}')
        
        ax.set_xlabel('Absolute Error (ppm)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name} Error Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error_distributions.png")

def plot_errors_by_residue(results, cs_cols, save_dir):
    """Error analysis by residue type"""
    print("\nCreating error by residue plots...")
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
        mask = results['masks'][:, i]
        errors = results['errors'][mask, i]
        residues = results['residues'][mask]
        
        # Create dataframe
        df = pd.DataFrame({
            'residue': residues,
            'error': errors
        })
        
        # Calculate stats per residue
        residue_stats = df.groupby('residue')['error'].agg(['mean', 'median', 'std', 'count']).reset_index()
        residue_stats = residue_stats.sort_values('mean')
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Box plot
        residue_order = residue_stats['residue'].values
        df_plot = df[df['residue'].isin(residue_order)]
        
        sns.boxplot(data=df_plot, x='residue', y='error', order=residue_order, ax=ax)
        
        # Add count labels
        for j, res in enumerate(residue_order):
            count = residue_stats[residue_stats['residue'] == res]['count'].values[0]
            ax.text(j, ax.get_ylim()[1] * 0.95, f'n={count}', ha='center', fontsize=8)
        
        ax.set_xlabel('Residue Type', fontsize=12)
        ax.set_ylabel('Absolute Error (ppm)', fontsize=12)
        ax.set_title(f'{name} - Error by Residue Type', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/errors_by_residue_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved errors_by_residue_*.png")

def plot_errors_by_ss(results, cs_cols, save_dir):
    """Error analysis by secondary structure"""
    print("\nCreating error by secondary structure plots...")
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
        ax = axes[i]
        
        mask = results['masks'][:, i]
        errors = results['errors'][mask, i]
        ss_types = results['ss_types'][mask]
        
        # Create dataframe
        df = pd.DataFrame({
            'ss_type': ss_types,
            'error': errors
        })
        
        # Box plot
        sns.boxplot(data=df, x='ss_type', y='error', ax=ax)
        
        # Add statistics
        ss_stats = df.groupby('ss_type')['error'].agg(['mean', 'count']).reset_index()
        for j, ss in enumerate(ss_stats['ss_type']):
            mean_val = ss_stats[ss_stats['ss_type'] == ss]['mean'].values[0]
            count = ss_stats[ss_stats['ss_type'] == ss]['count'].values[0]
            ax.text(j, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.3f}\nn={count}', 
                   ha='center', fontsize=9)
        
        ax.set_xlabel('Secondary Structure Type', fontsize=12)
        ax.set_ylabel('Absolute Error (ppm)', fontsize=12)
        ax.set_title(f'{name} - Error by Secondary Structure', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/errors_by_ss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved errors_by_ss.png")

def plot_attention_patterns(results, save_dir):
    """Analyze and visualize attention patterns"""
    print("\nAnalyzing attention patterns...")
    
    attention = results['attention']  # [N, seq_len, 1]
    attention = attention.squeeze(-1)  # [N, seq_len]
    
    # Average attention across all samples
    avg_attention = np.mean(attention, axis=0)
    std_attention = np.std(attention, axis=0)
    
    # Plot average attention pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = np.arange(len(avg_attention)) - len(avg_attention)//2  # Center at 0
    
    ax.bar(positions, avg_attention, yerr=std_attention, alpha=0.7, capsize=5, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Center (prediction target)')
    
    ax.set_xlabel('Position Relative to Center', fontsize=14)
    ax.set_ylabel('Average Attention Weight', fontsize=14)
    ax.set_title('Attention Pattern Across Sequence Window\n(Averaged over all predictions)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (pos, att) in enumerate(zip(positions, avg_attention)):
        ax.text(pos, att + std_attention[i] + 0.01, f'{att*100:.1f}%', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved attention_pattern.png")
    
    # Attention by residue type
    plot_attention_by_residue(results, save_dir)
    
    # Attention by SS type
    plot_attention_by_ss(results, save_dir)
    
    # Heatmap of individual samples
    plot_attention_heatmap(results, save_dir)

def plot_attention_by_residue(results, save_dir, n_residues=10):
    """Attention patterns for different residue types"""
    print("\nAnalyzing attention by residue type...")
    
    attention = results['attention'].squeeze(-1)
    residues = results['residues']
    
    # Get most common residues
    unique_residues, counts = np.unique(residues, return_counts=True)
    top_residues = unique_residues[np.argsort(counts)[-n_residues:]]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    positions = np.arange(attention.shape[1]) - attention.shape[1]//2
    
    for i, res in enumerate(top_residues):
        ax = axes[i]
        
        mask = residues == res
        res_attention = attention[mask]
        
        avg_att = np.mean(res_attention, axis=0)
        std_att = np.std(res_attention, axis=0)
        
        ax.bar(positions, avg_att, yerr=std_att, alpha=0.7, capsize=3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        
        ax.set_title(f'{res} (n={mask.sum()})', fontsize=10)
        ax.set_xlabel('Position', fontsize=9)
        ax.set_ylabel('Attention', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Patterns by Residue Type', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_by_residue.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved attention_by_residue.png")

def plot_attention_by_ss(results, save_dir):
    """Attention patterns for different SS types"""
    print("\nAnalyzing attention by secondary structure...")
    
    attention = results['attention'].squeeze(-1)
    ss_types = results['ss_types']
    
    unique_ss = np.unique(ss_types)
    
    fig, axes = plt.subplots(1, len(unique_ss), figsize=(5*len(unique_ss), 5))
    if len(unique_ss) == 1:
        axes = [axes]
    
    positions = np.arange(attention.shape[1]) - attention.shape[1]//2
    
    for i, ss in enumerate(unique_ss):
        ax = axes[i]
        
        mask = ss_types == ss
        ss_attention = attention[mask]
        
        avg_att = np.mean(ss_attention, axis=0)
        std_att = np.std(ss_attention, axis=0)
        
        ax.bar(positions, avg_att, yerr=std_att, alpha=0.7, capsize=5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'SS Type: {ss} (n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Position Relative to Center', fontsize=11)
        ax.set_ylabel('Average Attention Weight', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Patterns by Secondary Structure', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_by_ss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved attention_by_ss.png")

def plot_attention_heatmap(results, save_dir, n_samples=100):
    """Heatmap of attention weights for individual samples"""
    print("\nCreating attention heatmap...")
    
    attention = results['attention'].squeeze(-1)
    
    # Sample random subset
    if len(attention) > n_samples:
        idx = np.random.choice(len(attention), n_samples, replace=False)
        attention_sample = attention[idx]
        residues_sample = results['residues'][idx]
    else:
        attention_sample = attention
        residues_sample = results['residues']
    
    # Sort by center attention (most interesting patterns first)
    center_idx = attention_sample.shape[1] // 2
    center_attention = attention_sample[:, center_idx]
    sort_idx = np.argsort(center_attention)[::-1]
    
    attention_sorted = attention_sample[sort_idx]
    residues_sorted = residues_sample[sort_idx]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 16))
    
    im = ax.imshow(attention_sorted, aspect='auto', cmap='YlOrRd')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)
    
    # X-axis: positions
    positions = np.arange(attention_sorted.shape[1]) - attention_sorted.shape[1]//2
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions)
    ax.set_xlabel('Position Relative to Center', fontsize=12)
    
    # Y-axis: samples with residue labels
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title('Attention Weights Heatmap\n(100 random samples, sorted by center attention)', fontsize=14)
    
    # Add vertical line at center
    ax.axvline(center_idx, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved attention_heatmap.png")

def analyze_conv_features(model, loader, device, save_dir):
    """Analyze conv feature statistics at each position"""
    print("\nAnalyzing conv feature statistics by position...")
    model.eval()
    
    position_features = [[] for _ in range(9)]  # 9 positions in window
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting conv features'):
            features = batch['features'].to(device)
            
            # Get conv output (before attention)
            x = features.transpose(1, 2)
            x = model.conv(x)
            x = x.transpose(1, 2)  # [batch, seq_len, 512]
            
            # Collect features for each position
            for pos in range(9):
                position_features[pos].append(x[:, pos, :].cpu().numpy())
    
    # Concatenate
    for pos in range(9):
        position_features[pos] = np.concatenate(position_features[pos], axis=0)
    
    # Analyze statistics
    print("\n" + "="*60)
    print("CONV FEATURE STATISTICS BY POSITION")
    print("="*60)
    
    for pos in range(9):
        feat = position_features[pos]
        mean_norm = np.mean(np.linalg.norm(feat, axis=1))
        mean_val = np.mean(feat)
        std_val = np.std(feat)
        
        print(f"Position {pos} (relative: {pos-4}):")
        print(f"  Mean L2 norm: {mean_norm:.3f}")
        print(f"  Mean value:   {mean_val:.3f}")
        print(f"  Std dev:      {std_val:.3f}")
    
    # Plot feature magnitudes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = np.arange(9) - 4
    norms = [np.mean(np.linalg.norm(position_features[pos], axis=1)) 
             for pos in range(9)]
    
    ax.bar(positions, norms, alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Center (prediction target)')
    ax.set_xlabel('Position Relative to Center', fontsize=12)
    ax.set_ylabel('Average Feature L2 Norm', fontsize=12)
    ax.set_title('Conv Feature Magnitude by Position\n(Higher = stronger features)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (pos, norm) in enumerate(zip(positions, norms)):
        ax.text(pos, norm + max(norms)*0.02, f'{norm:.1f}', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conv_feature_magnitudes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved conv_feature_magnitudes.png")
    
    return position_features

def analyze_attention_scores(model, loader, device, save_dir):
    """Analyze attention scores before softmax"""
    print("\nAnalyzing attention scores (pre-softmax)...")
    model.eval()
    
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting attention scores'):
            features = batch['features'].to(device)
            
            # Get conv output
            x = features.transpose(1, 2)
            x = model.conv(x)
            x = x.transpose(1, 2)
            
            # Get attention SCORES (before softmax)
            attn_scores = model.attn(x)  # [batch, seq_len, 1]
            all_scores.append(attn_scores.squeeze(-1).cpu().numpy())
    
    all_scores = np.concatenate(all_scores, axis=0)
    
    # Analyze
    print("\n" + "="*60)
    print("ATTENTION SCORES (BEFORE SOFTMAX)")
    print("="*60)
    
    for pos in range(9):
        scores = all_scores[:, pos]
        print(f"Position {pos} (relative: {pos-4}):")
        print(f"  Mean score: {np.mean(scores):7.3f}")
        print(f"  Std score:  {np.std(scores):7.3f}")
        print(f"  Min score:  {np.min(scores):7.3f}")
        print(f"  Max score:  {np.max(scores):7.3f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    positions = np.arange(9) - 4
    mean_scores = [np.mean(all_scores[:, pos]) for pos in range(9)]
    std_scores = [np.std(all_scores[:, pos]) for pos in range(9)]
    
    # Plot 1: Mean scores
    ax1.bar(positions, mean_scores, yerr=std_scores, alpha=0.7, capsize=5, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Center (prediction target)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Position Relative to Center', fontsize=12)
    ax1.set_ylabel('Attention Score (pre-softmax)', fontsize=12)
    ax1.set_title('Attention Scores Before Softmax\n(Higher score = more likely to win softmax)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (pos, score) in enumerate(zip(positions, mean_scores)):
        ax1.text(pos, score + std_scores[i] + max(mean_scores)*0.05, f'{score:.2f}', 
               ha='center', fontsize=10)
    
    # Plot 2: Distribution of scores for each position
    box_data = [all_scores[:, pos] for pos in range(9)]
    bp = ax2.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Center')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Position Relative to Center', fontsize=12)
    ax2.set_ylabel('Attention Score Distribution', fontsize=12)
    ax2.set_title('Score Distributions Across Positions', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_scores_presoftmax.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved attention_scores_presoftmax.png")
    
    return all_scores

def analyze_data_correlation(test_df, save_dir):
    """Check if there's correlation between position i and i-4 in the data"""
    print("\nChecking data correlation between position i and i-4...")
    
    correlations = {col: [] for col in ['CS1', 'CS2', 'CS3', 'CS4']}
    
    protein_sample = test_df['protein_id'].unique()[:100]  # Sample 100 proteins
    
    for protein_id in tqdm(protein_sample, desc='Analyzing proteins'):
        prot_df = test_df[test_df['protein_id'] == protein_id].sort_values('position')
        
        if len(prot_df) < 5:
            continue
        
        for i in range(4, len(prot_df)):
            for col in ['CS1', 'CS2', 'CS3', 'CS4']:
                val_i = prot_df.iloc[i][col]
                val_i_minus_4 = prot_df.iloc[i-4][col]
                
                if val_i != 9999 and val_i_minus_4 != 9999:
                    correlations[col].append((val_i, val_i_minus_4))
    
    # Calculate and plot correlations
    print("\n" + "="*60)
    print("DATA CORRELATION: Position i vs Position i-4")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    for idx, (col, name) in enumerate(zip(['CS1', 'CS2', 'CS3', 'CS4'], cs_names)):
        ax = axes[idx]
        
        if correlations[col]:
            vals = np.array(correlations[col])
            corr = np.corrcoef(vals[:, 0], vals[:, 1])[0, 1]
            
            print(f"{col:8s}: Correlation = {corr:6.3f} (n={len(vals):,})")
            
            # Scatter plot with density
            if len(vals) > 1000:
                sample_idx = np.random.choice(len(vals), 1000, replace=False)
                x, y = vals[sample_idx, 1], vals[sample_idx, 0]
            else:
                x, y = vals[:, 1], vals[:, 0]
            
            ax.scatter(x, y, alpha=0.3, s=20)
            
            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit')
            
            ax.set_xlabel(f'{name} at position i-4 (ppm)', fontsize=11)
            ax.set_ylabel(f'{name} at position i (ppm)', fontsize=11)
            ax.set_title(f'{name}\nCorrelation = {corr:.3f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_correlation_i_vs_i-4.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved data_correlation_i_vs_i-4.png")
    print("="*60)

def plot_correlation_matrix(results, cs_cols, save_dir):
    """Correlation between different CS types"""
    print("\nCreating CS correlation matrix...")
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    # Get valid predictions for each CS type
    correlations = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            mask = results['masks'][:, i] & results['masks'][:, j]
            if mask.sum() > 0:
                pred_i = results['predictions'][mask, i]
                pred_j = results['predictions'][mask, j]
                correlations[i, j] = np.corrcoef(pred_i, pred_j)[0, 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Labels
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(cs_names, rotation=45, ha='right')
    ax.set_yticklabels(cs_names)
    
    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{correlations[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=12)
    
    ax.set_title('Chemical Shift Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cs_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved cs_correlation.png")

def plot_worst_predictions(results, cs_cols, save_dir, n=20):
    """Analyze worst predictions"""
    print("\nAnalyzing worst predictions...")
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
        mask = results['masks'][:, i]
        errors = results['errors'][mask, i]
        predictions = results['predictions'][mask, i]
        targets = results['targets'][mask, i]
        residues = results['residues'][mask]
        ss_types = results['ss_types'][mask]
        
        # Get worst predictions
        worst_idx = np.argsort(errors)[-n:][::-1]
        
        # Create dataframe
        df = pd.DataFrame({
            'Error': errors[worst_idx],
            'Predicted': predictions[worst_idx],
            'Target': targets[worst_idx],
            'Residue': residues[worst_idx],
            'SS Type': ss_types[worst_idx]
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Predicted'], width, label='Predicted', alpha=0.7)
        ax.bar(x + width/2, df['Target'], width, label='Target', alpha=0.7)
        
        # Add error annotations
        for j, (pred, target, err, res, ss) in enumerate(zip(df['Predicted'], df['Target'], df['Error'], df['Residue'], df['SS Type'])):
            ax.text(j, max(pred, target) + 5, f'Δ={err:.2f}\n{res}\n{ss}', 
                   ha='center', fontsize=8)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_title(f'{name} - Top {n} Worst Predictions', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in range(len(df))], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/worst_predictions_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved worst_predictions_*.png")

def plot_error_vs_distance(results, cs_cols, save_dir):
    """Error vs distance metrics (phi, psi angles)"""
    print("\nAnalyzing error vs structural features...")
    
    # This would require loading the actual phi/psi values from the dataset
    # For now, we'll skip this or implement if needed
    pass

def create_summary_report(results, cs_cols, stats, save_dir):
    """Create a text summary report"""
    print("\nCreating summary report...")
    
    cs_names = ['N (CS1)', 'H-N (CS2)', 'C-α (CS3)', 'H-α (CS4)']
    
    with open(f'{save_dir}/summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHEMICAL SHIFT PREDICTION MODEL - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        
        for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
            mask = results['masks'][:, i]
            errors = results['errors'][mask, i]
            predictions = results['predictions'][mask, i]
            targets = results['targets'][mask, i]
            
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean((predictions - targets)**2))
            median_error = np.median(errors)
            p95_error = np.percentile(errors, 95)
            r2 = np.corrcoef(predictions, targets)[0, 1]**2
            
            f.write(f"\n{name}:\n")
            f.write(f"  MAE:              {mae:.3f} ppm\n")
            f.write(f"  RMSE:             {rmse:.3f} ppm\n")
            f.write(f"  Median Error:     {median_error:.3f} ppm\n")
            f.write(f"  95th Percentile:  {p95_error:.3f} ppm\n")
            f.write(f"  R²:               {r2:.3f}\n")
            f.write(f"  N predictions:    {mask.sum()}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR BY RESIDUE TYPE\n")
        f.write("-"*80 + "\n")
        
        for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
            mask = results['masks'][:, i]
            errors = results['errors'][mask, i]
            residues = results['residues'][mask]
            
            df = pd.DataFrame({'residue': residues, 'error': errors})
            stats_df = df.groupby('residue')['error'].agg(['mean', 'std', 'count']).sort_values('mean')
            
            f.write(f"\n{name}:\n")
            f.write(stats_df.to_string())
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR BY SECONDARY STRUCTURE\n")
        f.write("-"*80 + "\n")
        
        for i, (col, name) in enumerate(zip(cs_cols, cs_names)):
            mask = results['masks'][:, i]
            errors = results['errors'][mask, i]
            ss_types = results['ss_types'][mask]
            
            df = pd.DataFrame({'ss_type': ss_types, 'error': errors})
            stats_df = df.groupby('ss_type')['error'].agg(['mean', 'std', 'count']).sort_values('mean')
            
            f.write(f"\n{name}:\n")
            f.write(stats_df.to_string())
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ATTENTION ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        attention = results['attention'].squeeze(-1)
        avg_attention = np.mean(attention, axis=0)
        positions = np.arange(len(avg_attention)) - len(avg_attention)//2
        
        f.write("\nAverage Attention Distribution:\n")
        for pos, att in zip(positions, avg_attention):
            f.write(f"  Position {pos:+2d}: {att*100:5.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved summary_report.txt")

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL ANALYSIS - Deep Dive into Your Chemical Shift Predictor")
    print("="*80)
    
    # Load data
    train_df, test_df = load_and_split_data(DATA_PATH)
    stats = compute_stats(train_df)
    cs_cols = ['CS1', 'CS2', 'CS3', 'CS4']
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = ProteinDataset(train_df, stats, context_window=CONTEXT_WINDOW//2, 
                                   mask_prob=MASK_PROB, training=True)
    
    test_dataset = ProteinDataset(test_df, stats, context_window=CONTEXT_WINDOW//2, 
                                  mask_prob=MASK_PROB, training=False,
                                  residue_encoder=train_dataset.residue_encoder,
                                  ss_encoder=train_dataset.ss_encoder,
                                  seed=42)
    
    # Create data loaders
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    # Load model
    print("\nLoading model...")
    in_feat = train_dataset.n_residue_types + train_dataset.n_ss_types + 4 + 3
    model = ChemicalShiftCNN(in_feat=in_feat, channels=[128, 256, 512], 
                            kernel=3, dropout=0.3)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Run comprehensive analysis
    save_dir = 'analysis_viz'
    print(f"\nRunning comprehensive analysis...")
    print(f"Results will be saved to: {save_dir}/")
    
    results = analyze_predictions(model, test_loader, DEVICE, stats, cs_cols, save_dir)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create all visualizations
    plot_prediction_scatter(results, cs_cols, save_dir)
    plot_error_distributions(results, cs_cols, save_dir)
    plot_errors_by_residue(results, cs_cols, save_dir)
    plot_errors_by_ss(results, cs_cols, save_dir)
    plot_attention_patterns(results, save_dir)
    plot_correlation_matrix(results, cs_cols, save_dir)
    plot_worst_predictions(results, cs_cols, save_dir)
    
    # NEW DEBUG ANALYSES
    print("\n" + "="*80)
    print("DEEP DIVE: ATTENTION MECHANISM DEBUG")
    print("="*80)
    analyze_conv_features(model, test_loader, DEVICE, save_dir)
    analyze_attention_scores(model, test_loader, DEVICE, save_dir)
    analyze_data_correlation(test_df, save_dir)
    
    create_summary_report(results, cs_cols, stats, save_dir)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {save_dir}/")
    print("\nGenerated files:")
    
    for file in sorted(os.listdir(save_dir)):
        if file.endswith('.png') or file.endswith('.txt'):
            print(f"  • {file}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()