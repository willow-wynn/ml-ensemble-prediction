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

DATA_DIR = '/Users/wynndiaz//Data'
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

# EDL-specific parameters
# Two-stage training with gradual transition:
# Stage 1 (0-15): Pure CE (learn predictions)
# Stage 2 (16-30): Blend CE→EDL (smooth transition)
# Stage 3 (31+): Pure EDL (uncertainty quantification)
CE_WARMUP_EPOCHS = 15  # Use CE for first N epochs
BLEND_EPOCHS = 15  # Epochs to blend from CE to EDL
PHI_BINS = 36  # 10° bins for phi (-180° to 180°)
PSI_BINS = 36  # 10° bins for psi (-180° to 180°)
NUM_CLASSES = PHI_BINS * PSI_BINS  # 1296 total bins
EDL_LOSS_SCALE = 1000.0  # Scale EDL loss to match CE magnitude
EDL_LOSS_CLAMP = 5.0  # Clamp pure EDL loss to prevent explosions
EDL_LR_MULTIPLIER = 30.0  # Boost LR by 3× during pure EDL phase
LAMBDA_COEF = 0.01  # KL divergence weight
ANNEALING_STEP = 10  # Epochs to reach full KL weight

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== Utility Functions ====================

def angles_to_bin(phi, psi, phi_bins=PHI_BINS, psi_bins=PSI_BINS):
    """
    Convert continuous angles to discrete bin indices.
    
    Args:
        phi: angle in radians (-π to π)
        psi: angle in radians (-π to π)
    
    Returns:
        bin_idx: integer in [0, phi_bins * psi_bins - 1]
    """
    # Convert to degrees and shift to [0, 360)
    phi_deg = np.rad2deg(phi) + 180
    psi_deg = np.rad2deg(psi) + 180
    
    # Discretize
    phi_bin = int(phi_deg * phi_bins / 360.0)
    psi_bin = int(psi_deg * psi_bins / 360.0)
    
    # Clamp to valid range
    phi_bin = min(phi_bin, phi_bins - 1)
    psi_bin = min(psi_bin, psi_bins - 1)
    
    # Flatten to single index
    return phi_bin * psi_bins + psi_bin


def bin_to_angles(bin_idx, phi_bins=PHI_BINS, psi_bins=PSI_BINS):
    """
    Convert bin index back to angle (bin center).
    
    Returns:
        phi, psi: angles in radians
    """
    phi_bin = bin_idx // psi_bins
    psi_bin = bin_idx % psi_bins
    
    # Get bin centers
    phi_deg = (phi_bin + 0.5) * 360.0 / phi_bins - 180
    psi_deg = (psi_bin + 0.5) * 360.0 / psi_bins - 180
    
    return np.deg2rad(phi_deg), np.deg2rad(psi_deg)


def bin_to_angles_batch(bin_indices, phi_bins=PHI_BINS, psi_bins=PSI_BINS):
    """
    Convert batch of bin indices to angles.
    
    Args:
        bin_indices: (batch,) array of bin indices
    
    Returns:
        phi, psi: (batch,) arrays of angles in radians
    """
    phi_bins_idx = bin_indices // psi_bins
    psi_bins_idx = bin_indices % psi_bins
    
    phi_deg = (phi_bins_idx + 0.5) * 360.0 / phi_bins - 180
    psi_deg = (psi_bins_idx + 0.5) * 360.0 / psi_bins - 180
    
    return np.deg2rad(phi_deg), np.deg2rad(psi_deg)


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

    # Drop rows with NaN values first
    before = len(df)
    df = df.dropna()
    if before - len(df) > 0:
        print(f"Dropped {before - len(df)} rows with NaN values")

    # Create mask columns BEFORE replacing 9999
    for col in ['H', 'N', 'HA', 'CA', 'Cp']:
        mask_col = f'{col}_mask'
        df[mask_col] = (df[col] != 9999.0).astype(np.float32)

    # Replace 9999 with 0
    for col in ['H', 'N', 'HA', 'CA', 'Cp']:
        df[col] = df[col].replace(9999.0, 0.0)

    # Create mask for valid angles
    df['angles_valid'] = ((df['PHI'].abs() < 360) & (df['PSI'].abs() < 360)).astype(np.float32)
    invalid_count = (df['angles_valid'] == 0).sum()
    print(f"Rows with invalid angles (360°): {invalid_count}")

    # Replace 360 angles with 0
    df['PHI'] = df['PHI'].replace(360.0, 0.0).replace(-360.0, 0.0)
    df['PSI'] = df['PSI'].replace(360.0, 0.0).replace(-360.0, 0.0)

    print(f"Final dataset: {df.shape}")
    return df


# ==================== Dataset ====================

class ContextWindowDataset(Dataset):
    """Dataset with context window - returns bin labels for EDL"""

    def __init__(self, df, context_window=3, phi_bins=PHI_BINS, psi_bins=PSI_BINS):
        self.df = df.reset_index(drop=True)
        self.context_window = context_window
        self.phi_bins = phi_bins
        self.psi_bins = psi_bins
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

        # Only keep valid samples
        self.valid_indices = []
        for i in range(context_window, len(df) - context_window):
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

        # Safety check
        if abs(phi_deg) >= 360 or abs(psi_deg) >= 360:
            raise ValueError(f"Invalid angle in dataset! PHI={phi_deg}, PSI={psi_deg} at index {center_idx}")

        phi_rad = np.deg2rad(phi_deg)
        psi_rad = np.deg2rad(psi_deg)

        # Convert to bin index
        bin_idx = angles_to_bin(phi_rad, psi_rad, self.phi_bins, self.psi_bins)

        # Also return continuous angles for evaluation
        angles = np.array([phi_rad, psi_rad], dtype=np.float32)

        # Validate no NaN/Inf
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError(f"NaN/Inf in features at index {center_idx}")
        if np.isnan(angles).any() or np.isinf(angles).any():
            raise ValueError(f"NaN/Inf in angles at index {center_idx}")

        return (torch.FloatTensor(features),
                torch.LongTensor([bin_idx]),
                torch.FloatTensor(angles))


# ==================== Model Architecture ====================

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


class EvidentialCNN(nn.Module):
    """
    CNN that outputs evidence for Dirichlet distribution over angle bins.
    Based on EDL paper (Sensoy et al. 2018).
    """

    def __init__(self, in_feat=30, channels=[64, 128, 256], kernel=3,
                 dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

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

        # Evidence head - outputs non-negative evidence values
        self.evidence_head = nn.Sequential(
            nn.Linear(channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize final layer to have positive bias (start with some evidence)
        # This prevents immediate collapse to uniform distribution
        nn.init.xavier_normal_(self.evidence_head[-1].weight, gain=0.1)
        nn.init.constant_(self.evidence_head[-1].bias, 2.0)  # Start with evidence ≈ 2

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feat)

        Returns:
            evidence: (batch, num_classes) - non-negative evidence values
            alpha: (batch, num_classes) - Dirichlet parameters (evidence + 1)
            prob: (batch, num_classes) - expected probabilities
            uncertainty: (batch,) - total uncertainty (K/S)
        """
        # Extract features
        x = x.transpose(1, 2)  # (batch, feat, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # Attention pooling
        attn_w = F.softmax(self.attn(x), dim=1)
        x = torch.sum(x * attn_w, dim=1)  # (batch, channels)

        # Get evidence (use softplus for smooth non-negative activation)
        logits = self.evidence_head(x)  # (batch, num_classes)
        evidence = F.softplus(logits)  # Smooth, always positive, better gradients than ReLU

        # Dirichlet parameters: α = evidence + 1
        alpha = evidence + 1

        # Dirichlet strength
        S = torch.sum(alpha, dim=1, keepdim=True)  # (batch, 1)

        # Expected probabilities (mean of Dirichlet)
        prob = alpha / S  # (batch, num_classes)

        # Total uncertainty: K / S
        # When S = K (all evidence = 0), uncertainty = 1 (maximum)
        # When S >> K (lots of evidence), uncertainty -> 0
        uncertainty = self.num_classes / S.squeeze(1)  # (batch,)

        return evidence, alpha, prob, uncertainty


# ==================== EDL Loss Function ====================

class EDLLoss(nn.Module):
    """
    Evidential Deep Learning loss with three-stage training:
    Stage 1 (epoch <= ce_warmup): Pure CE for learning predictions
    Stage 2 (ce_warmup < epoch <= ce_warmup + blend): Gradual CE→EDL blend
    Stage 3 (epoch > ce_warmup + blend): Pure EDL (clamped to prevent explosions)
    
    Note: EDL loss is scaled up (default 1000×) to match CE loss magnitude.
    Without scaling, MSE/K with K=1296 gives loss ~0.0008, while CE gives ~5.0.
    Scaling ensures meaningful gradients during blending and pure EDL stages.
    
    Stage 3 clamps loss to max 5.0 to prevent training instability.
    """

    def __init__(self, num_classes=NUM_CLASSES, lambda_coef=LAMBDA_COEF,
                 annealing_step=ANNEALING_STEP, ce_warmup_epochs=15, blend_epochs=15,
                 edl_loss_scale=1000.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coef = lambda_coef
        self.annealing_step = annealing_step
        self.ce_warmup_epochs = ce_warmup_epochs
        self.blend_epochs = blend_epochs
        self.edl_loss_scale = edl_loss_scale  # Scale EDL to match CE magnitude
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, evidence, alpha, target, epoch):
        """
        Args:
            evidence: (batch, K) - evidence values
            alpha: (batch, K) - Dirichlet parameters
            target: (batch,) - true class indices
            epoch: current epoch for stage selection

        Returns:
            loss: scalar
            loss_dict: dictionary with loss components
        """
        batch_size = evidence.size(0)
        K = self.num_classes

        # Compute Dirichlet strength and probabilities (needed for all stages)
        S = torch.sum(alpha, dim=1, keepdim=True)  # (batch, 1)
        p = alpha / S  # (batch, K)

        # Stage 1: Pure CE warmup
        if epoch <= self.ce_warmup_epochs:
            logits = torch.log(p + 1e-10)  # Log probabilities
            loss_ce = self.ce_loss(logits, target.squeeze(1))
            
            return loss_ce, {
                'loss_ce': loss_ce.item(),
                'loss_mse': 0.0,
                'loss_kl': 0.0,
                'total_loss': loss_ce.item(),
                'annealing_coef': 0.0,
                'blend_coef': 0.0,
                'stage': 'CE_WARMUP'
            }

        # Compute EDL loss components (needed for stages 2 & 3)
        # Convert target to one-hot
        y = torch.zeros(batch_size, K, device=evidence.device)
        y.scatter_(1, target, 1)  # (batch, K)

        # Sum of Squares Loss (Equation 5)
        err = (y - p) ** 2
        var = p * (1 - p) / (S + 1)
        loss_mse = torch.mean(torch.sum(err + var, dim=1) / K)

        # KL Divergence
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(K), device=evidence.device))
        kl = kl - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        kl = kl + torch.sum(
            (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1, keepdim=True
        )
        kl = kl.squeeze(1)
        loss_kl = torch.mean(kl)

        # KL annealing (for EDL component)
        edl_epoch = epoch - self.ce_warmup_epochs
        if edl_epoch <= 5:
            annealing_coef = 0.0
        else:
            annealing_coef = min(1.0, (edl_epoch - 5) / self.annealing_step)

        # Scale EDL loss to match CE magnitude (CE ~5.0, unscaled MSE ~0.0008)
        loss_edl = self.edl_loss_scale * (loss_mse + annealing_coef * self.lambda_coef * loss_kl)

        # Stage 2: Gradual blend from CE to EDL
        if epoch <= self.ce_warmup_epochs + self.blend_epochs:
            # Compute CE loss
            logits = torch.log(p + 1e-10)
            loss_ce = self.ce_loss(logits, target.squeeze(1))
            
            # Smooth blend: 0→1 over blend_epochs
            blend_coef = (epoch - self.ce_warmup_epochs) / self.blend_epochs
            loss = (1 - blend_coef) * loss_ce + blend_coef * loss_edl
            
            return loss, {
                'loss_ce': loss_ce.item(),
                'loss_mse': loss_mse.item(),
                'loss_kl': loss_kl.item(),
                'total_loss': loss.item(),
                'annealing_coef': annealing_coef,
                'blend_coef': blend_coef,
                'stage': 'BLENDING'
            }

        # Stage 3: Pure EDL
        # Clamp loss to prevent explosions during pure EDL learning
        loss_edl_clamped = torch.clamp(loss_edl, max=5.0)
        
        return loss_edl_clamped, {
            'loss_ce': 0.0,
            'loss_mse': loss_mse.item(),
            'loss_kl': loss_kl.item(),
            'total_loss': loss_edl_clamped.item(),
            'unclamped_loss': loss_edl.item(),  # Track if clamping occurred
            'annealing_coef': annealing_coef,
            'blend_coef': 1.0,
            'stage': 'EDL'
        }


# ==================== Training ====================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    grad_norms = []
    batch_times = []
    uncertainties = []
    start_time = time.time()

    for batch_idx, (feat, bin_labels, angles) in enumerate(loader):
        batch_start = time.time()
        feat = feat.to(device)
        bin_labels = bin_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        evidence, alpha, prob, uncertainty = model(feat)

        # Compute loss
        loss, loss_dict = criterion(evidence, alpha, bin_labels, epoch)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"\n!!! NaN loss detected at batch {batch_idx}")
            print(f"Evidence range: [{evidence.min().item():.4f}, {evidence.max().item():.4f}]")
            print(f"Alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
            print(f"Prob range: [{prob.min().item():.4f}, {prob.max().item():.4f}]")
            raise ValueError("NaN loss encountered - stopping training")

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Check for NaN gradients
        if torch.isnan(grad_norm):
            print(f"\n!!! NaN gradients at batch {batch_idx}")
            print(f"Loss was: {loss.item():.4f}")
            raise ValueError("NaN gradients encountered - stopping training")

        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        grad_norms.append(grad_norm.item())
        batch_times.append(time.time() - batch_start)
        uncertainties.extend(uncertainty.detach().cpu().numpy())

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg = running_loss / LOG_INTERVAL
            avg_time = np.mean(batch_times[-LOG_INTERVAL:])
            samples_per_sec = BATCH_SIZE / avg_time
            avg_uncertainty = np.mean(uncertainties[-BATCH_SIZE*LOG_INTERVAL:])
            print(f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}] '
                  f'Loss: {avg:.4f} | GradNorm: {grad_norm:.4f} | '
                  f'Uncertainty: {avg_uncertainty:.4f} | {samples_per_sec:.0f} samples/s')
            running_loss = 0.0

    epoch_time = time.time() - start_time
    return {
        'loss': total_loss / len(loader),
        'grad_norm_mean': np.mean(grad_norms),
        'grad_norm_std': np.std(grad_norms),
        'grad_norm_max': np.max(grad_norms),
        'uncertainty_mean': np.mean(uncertainties),
        'uncertainty_std': np.std(uncertainties),
        'epoch_time': epoch_time,
        'samples_per_sec': len(loader.dataset) / epoch_time,
        'avg_batch_time': np.mean(batch_times)
    }


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    start_time = time.time()

    all_prob = []
    all_bin_labels = []
    all_angles = []
    all_uncertainty = []
    all_evidence = []

    with torch.no_grad():
        for feat, bin_labels, angles in loader:
            feat = feat.to(device)
            bin_labels = bin_labels.to(device)

            # Forward pass
            evidence, alpha, prob, uncertainty = model(feat)

            # Compute loss
            loss, _ = criterion(evidence, alpha, bin_labels, epoch)
            total_loss += loss.item()

            # Store predictions
            all_prob.append(prob.cpu().numpy())
            all_bin_labels.append(bin_labels.cpu().numpy())
            all_angles.append(angles.cpu().numpy())
            all_uncertainty.append(uncertainty.cpu().numpy())
            all_evidence.append(evidence.cpu().numpy())

    eval_time = time.time() - start_time

    # Concatenate all batches
    prob = np.vstack(all_prob)  # (N, 1296)
    bin_labels = np.concatenate(all_bin_labels).flatten()  # (N,)
    angles = np.vstack(all_angles)  # (N, 2)
    uncertainty = np.concatenate(all_uncertainty)  # (N,)
    evidence = np.vstack(all_evidence)  # (N, 1296)

    # Get predicted bins
    pred_bins = np.argmax(prob, axis=1)  # (N,)

    # Classification accuracy
    accuracy = np.mean(pred_bins == bin_labels)

    # Convert predicted bins to angles for MAE/RMSE
    pred_phi, pred_psi = bin_to_angles_batch(pred_bins)
    true_phi = angles[:, 0]
    true_psi = angles[:, 1]

    # Calculate circular errors
    phi_err = np.abs(np.angle(np.exp(1j * (true_phi - pred_phi))))
    psi_err = np.abs(np.angle(np.exp(1j * (true_psi - pred_psi))))

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

    # Uncertainty statistics
    uncertainty_mean = np.mean(uncertainty)
    uncertainty_median = np.median(uncertainty)
    uncertainty_std = np.std(uncertainty)

    # Correlation between uncertainty and error
    avg_err = (phi_err + psi_err) / 2
    uncertainty_error_corr = np.corrcoef(uncertainty, avg_err)[0, 1]

    # Evidence statistics
    total_evidence_mean = np.mean(np.sum(evidence, axis=1))
    max_evidence_mean = np.mean(np.max(evidence, axis=1))

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy * 100,
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
        'uncertainty_mean': uncertainty_mean,
        'uncertainty_median': uncertainty_median,
        'uncertainty_std': uncertainty_std,
        'uncertainty_error_corr': uncertainty_error_corr,
        'total_evidence_mean': total_evidence_mean,
        'max_evidence_mean': max_evidence_mean,
        'eval_time': eval_time,
        'pred_phi': pred_phi,
        'pred_psi': pred_psi,
        'true_phi': true_phi,
        'true_psi': true_psi,
        'uncertainty': uncertainty,
        'phi_err': phi_err,
        'psi_err': psi_err,
        'evidence': evidence,
        'prob': prob
    }


# ==================== Ensemble Generation ====================

def generate_ensemble(evidence, alpha, n_samples=100, seed=None):
    """
    Generate ensemble of angle predictions by sampling from Dirichlet distribution.
    
    Args:
        evidence: (batch, K) - evidence values
        alpha: (batch, K) - Dirichlet parameters
        n_samples: number of samples to generate per input
        seed: random seed for reproducibility
    
    Returns:
        ensemble_phi: (batch, n_samples) - phi angles in radians
        ensemble_psi: (batch, n_samples) - psi angles in radians
    """
    if seed is not None:
        np.random.seed(seed)
    
    batch_size = alpha.shape[0]
    K = alpha.shape[1]
    
    ensemble_phi = np.zeros((batch_size, n_samples))
    ensemble_psi = np.zeros((batch_size, n_samples))
    
    for i in range(batch_size):
        # Sample probability distributions from Dirichlet
        # Each sample is a different probability distribution over bins
        sampled_probs = np.random.dirichlet(alpha[i].cpu().numpy(), size=n_samples)  # (n_samples, K)
        
        # For each sampled probability distribution, sample a bin
        for j in range(n_samples):
            bin_idx = np.random.choice(K, p=sampled_probs[j])
            phi, psi = bin_to_angles(bin_idx)
            ensemble_phi[i, j] = phi
            ensemble_psi[i, j] = psi
    
    return ensemble_phi, ensemble_psi


# ==================== Visualization ====================

def plot_results(results, save_path='results_edl.png'):
    """Plot predictions vs targets with uncertainty"""
    pred_phi = results['pred_phi']
    pred_psi = results['pred_psi']
    true_phi = results['true_phi']
    true_psi = results['true_psi']
    uncertainty = results['uncertainty']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Phi predictions
    ax = axes[0, 0]
    scatter = ax.scatter(np.rad2deg(true_phi), np.rad2deg(pred_phi),
                        c=uncertainty, cmap='viridis', alpha=0.3, s=1)
    ax.plot([-180, 180], [-180, 180], 'r--', lw=2)
    ax.set_xlabel('True Phi (°)')
    ax.set_ylabel('Predicted Phi (°)')
    ax.set_title(f'Phi (MAE={results["phi_mae"]:.1f}°)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Uncertainty')
    
    # Psi predictions
    ax = axes[0, 1]
    scatter = ax.scatter(np.rad2deg(true_psi), np.rad2deg(pred_psi),
                        c=uncertainty, cmap='viridis', alpha=0.3, s=1)
    ax.plot([-180, 180], [-180, 180], 'r--', lw=2)
    ax.set_xlabel('True Psi (°)')
    ax.set_ylabel('Predicted Psi (°)')
    ax.set_title(f'Psi (MAE={results["psi_mae"]:.1f}°)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Uncertainty')
    
    # Ramachandran comparison
    ax = axes[0, 2]
    ax.hexbin(np.rad2deg(true_phi), np.rad2deg(true_psi),
              gridsize=50, cmap='Blues', alpha=0.6, mincnt=1)
    ax.hexbin(np.rad2deg(pred_phi), np.rad2deg(pred_psi),
              gridsize=50, cmap='Reds', alpha=0.4, mincnt=1)
    ax.set_xlabel('Phi (°)')
    ax.set_ylabel('Psi (°)')
    ax.set_title('Ramachandran (Blue=True, Red=Pred)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    # Uncertainty distribution
    ax = axes[1, 0]
    ax.hist(uncertainty, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(uncertainty), color='r', linestyle='--', label=f'Mean: {np.mean(uncertainty):.3f}')
    ax.axvline(np.median(uncertainty), color='g', linestyle='--', label=f'Median: {np.median(uncertainty):.3f}')
    ax.set_xlabel('Uncertainty (K/S)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Uncertainty vs Error
    ax = axes[1, 1]
    avg_err = (results['phi_err'] + results['psi_err']) / 2
    ax.hexbin(uncertainty, np.rad2deg(avg_err), gridsize=50, cmap='viridis', mincnt=1)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Average Error (°)')
    ax.set_title(f'Uncertainty vs Error (corr={results["uncertainty_error_corr"]:.3f})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax)
    
    # Classification accuracy by uncertainty
    ax = axes[1, 2]
    # Bin by uncertainty
    n_bins = 20
    uncertainty_bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins+1))
    bin_accuracy = []
    bin_centers = []
    for i in range(n_bins):
        mask = (uncertainty >= uncertainty_bins[i]) & (uncertainty < uncertainty_bins[i+1])
        if np.sum(mask) > 0:
            bin_err = avg_err[mask]
            bin_accuracy.append(np.mean(bin_err))
            bin_centers.append((uncertainty_bins[i] + uncertainty_bins[i+1]) / 2)
    
    ax.plot(bin_centers, np.rad2deg(bin_accuracy), 'o-', linewidth=2)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Average Error (°)')
    ax.set_title('Error vs Uncertainty (binned)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


def plot_ensemble_example(model, test_loader, device, n_examples=3, n_samples=100, 
                         save_path='ensemble_examples.png'):
    """
    Plot ensemble predictions for a few example residues.
    """
    model.eval()
    
    # Get first batch
    feat, bin_labels, angles = next(iter(test_loader))
    feat = feat.to(device)
    
    with torch.no_grad():
        evidence, alpha, prob, uncertainty = model(feat)
    
    # Generate ensembles for first n_examples
    ensemble_phi, ensemble_psi = generate_ensemble(
        evidence[:n_examples], alpha[:n_examples], n_samples=n_samples
    )
    
    # Get true and predicted angles
    true_phi = angles[:n_examples, 0].numpy()
    true_psi = angles[:n_examples, 1].numpy()
    pred_bins = torch.argmax(prob[:n_examples], dim=1).cpu().numpy()
    pred_phi, pred_psi = bin_to_angles_batch(pred_bins)
    uncertainties = uncertainty[:n_examples].cpu().numpy()
    
    fig, axes = plt.subplots(1, n_examples, figsize=(6*n_examples, 6))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(n_examples):
        ax = axes[i]
        
        # Plot ensemble samples
        ax.scatter(np.rad2deg(ensemble_phi[i]), np.rad2deg(ensemble_psi[i]),
                  alpha=0.3, s=20, c='blue', label='Ensemble')
        
        # Plot predicted (most likely)
        ax.scatter(np.rad2deg(pred_phi[i]), np.rad2deg(pred_psi[i]),
                  s=200, c='red', marker='*', edgecolors='black',
                  linewidths=2, label='Predicted', zorder=10)
        
        # Plot true
        ax.scatter(np.rad2deg(true_phi[i]), np.rad2deg(true_psi[i]),
                  s=200, c='green', marker='X', edgecolors='black',
                  linewidths=2, label='True', zorder=10)
        
        ax.set_xlabel('Phi (°)')
        ax.set_ylabel('Psi (°)')
        ax.set_title(f'Example {i+1} (Uncertainty={uncertainties[i]:.3f})')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Ensemble examples saved: {save_path}")
    plt.close()


# ==================== Main ====================

def main():
    print("="*80)
    print("EVIDENTIAL DEEP LEARNING FOR PROTEIN ANGLE PREDICTION")
    print("="*80)
    print(f"Architecture: 1D CNN with attention + EDL")
    print(f"Discretization: {PHI_BINS}x{PSI_BINS} = {NUM_CLASSES} bins")
    print(f"Loss: EDL (MSE + KL divergence)")
    print(f"Context: ±{CONTEXT_WINDOW} residues")
    print(f"Features: 20 AA + 5 chem shifts + 5 masks = 30 per residue")
    print(f"Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("="*80)

    # Load data
    df = load_and_preprocess_data(DATA_DIR, CSV_FILES)

    # Contiguous train-test split
    print("\n" + "="*80)
    print("Contiguous Train-Test Split")
    print("="*80)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {len(train_df)} residues (0 to {split_idx-1})")
    print(f"Test: {len(test_df)} residues ({split_idx} to {len(df)-1})")

    # Create datasets
    print("\n" + "="*80)
    print("Creating Datasets")
    print("="*80)
    train_ds = ContextWindowDataset(train_df, CONTEXT_WINDOW, PHI_BINS, PSI_BINS)
    test_ds = ContextWindowDataset(test_df, CONTEXT_WINDOW, PHI_BINS, PSI_BINS)

    # Optimized DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Model
    model = EvidentialCNN(in_feat=30, channels=CONV_CHANNELS,
                          kernel=KERNEL_SIZE, dropout=0.3,
                          num_classes=NUM_CLASSES).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    criterion = EDLLoss(num_classes=NUM_CLASSES, lambda_coef=LAMBDA_COEF,
                       annealing_step=ANNEALING_STEP, ce_warmup_epochs=CE_WARMUP_EPOCHS,
                       blend_epochs=BLEND_EPOCHS, edl_loss_scale=EDL_LOSS_SCALE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      patience=3, factor=0.5)

    print("\n" + "="*80)
    print("Training")
    print("="*80)
    print(f"Three-Stage Training:")
    print(f"  Stage 1 (Epochs 1-{CE_WARMUP_EPOCHS}): Pure CE - Learn to predict")
    print(f"  Stage 2 (Epochs {CE_WARMUP_EPOCHS+1}-{CE_WARMUP_EPOCHS+BLEND_EPOCHS}): Blend CE→EDL - Smooth transition")
    print(f"  Stage 3 (Epochs {CE_WARMUP_EPOCHS+BLEND_EPOCHS+1}+): Pure EDL - Uncertainty quantification")
    print(f"    • LR Boost: {EDL_LR_MULTIPLIER}× (from {LEARNING_RATE:.6f} to {LEARNING_RATE*EDL_LR_MULTIPLIER:.6f})")
    print(f"    • Loss Clamping: max {EDL_LOSS_CLAMP} to prevent explosions")
    print(f"Bins: {PHI_BINS}×{PSI_BINS} = {NUM_CLASSES} classes")
    print(f"EDL Loss Scale: {EDL_LOSS_SCALE}× (to match CE magnitude)")
    print("="*80)

    best = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---')

        train_stats = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        res = evaluate(model, test_loader, criterion, DEVICE, epoch)

        train_losses.append(train_stats['loss'])
        val_losses.append(res['loss'])

        scheduler.step(res['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Boost LR when entering pure EDL phase (epoch 31)
        if epoch == CE_WARMUP_EPOCHS + BLEND_EPOCHS:
            new_lr = LEARNING_RATE * EDL_LR_MULTIPLIER
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            current_lr = new_lr
            print(f"\n🚀 ENTERING PURE EDL PHASE - Boosting LR: {LEARNING_RATE:.6f} → {new_lr:.6f} ({EDL_LR_MULTIPLIER}×)")

        if res['loss'] < best:
            best = res['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': res['loss'],
                'phi_bins': PHI_BINS,
                'psi_bins': PSI_BINS,
                'num_classes': NUM_CLASSES
            }, 'glutton_edl_best.pth')
            print('  ★ Best model saved')

        # Comprehensive reporting
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{EPOCHS} SUMMARY")
        print(f"{'='*80}")

        # Loss metrics
        print(f"\nLoss Metrics:")
        stage = res.get('stage', 'UNKNOWN')
        if stage == 'CE_WARMUP':
            print(f"  Stage 1: CE WARMUP (learning to predict)")
            print(f"  Train Loss (CE): {train_stats['loss']:.4f} | Val Loss (CE): {res['loss']:.4f} | Best: {best:.4f}")
        elif stage == 'BLENDING':
            blend_coef = res.get('blend_coef', 0.0)
            print(f"  Stage 2: BLENDING (CE→EDL transition, blend={blend_coef:.2f})")
            print(f"  Train Loss: {train_stats['loss']:.4f} | Val Loss: {res['loss']:.4f} | Best: {best:.4f}")
        else:  # EDL
            print(f"  Stage 3: PURE EDL (uncertainty quantification)")
            print(f"  Train Loss: {train_stats['loss']:.4f} | Val Loss: {res['loss']:.4f} | Best: {best:.4f}")
            # Check if clamping occurred
            unclamped = res.get('unclamped_loss', res['loss'])
            if unclamped > EDL_LOSS_CLAMP:
                print(f"  ⚠️  Loss clamped: {unclamped:.4f} → {res['loss']:.4f}")

        # Classification metrics
        print(f"\nClassification:")
        print(f"  Accuracy: {res['accuracy']:.2f}%")

        # Angle prediction metrics
        print(f"\nAngle Prediction (degrees):")
        print(f"  Phi   - MAE: {res['phi_mae']:6.2f}° | RMSE: {res['phi_rmse']:6.2f}° | Median: {res['phi_median']:6.2f}° | 90th: {res['phi_p90']:6.2f}° | 95th: {res['phi_p95']:6.2f}°")
        print(f"  Psi   - MAE: {res['psi_mae']:6.2f}° | RMSE: {res['psi_rmse']:6.2f}° | Median: {res['psi_median']:6.2f}° | 90th: {res['psi_p90']:6.2f}° | 95th: {res['psi_p95']:6.2f}°")
        print(f"  Average MAE: {(res['phi_mae'] + res['psi_mae'])/2:.2f}°")

        # Uncertainty metrics
        print(f"\nUncertainty:")
        print(f"  Train - Mean: {train_stats['uncertainty_mean']:.4f} | Std: {train_stats['uncertainty_std']:.4f}")
        print(f"  Val   - Mean: {res['uncertainty_mean']:.4f} | Median: {res['uncertainty_median']:.4f} | Std: {res['uncertainty_std']:.4f}")
        print(f"  Uncertainty-Error Correlation: {res['uncertainty_error_corr']:.3f}")

        # Evidence metrics
        print(f"\nEvidence:")
        print(f"  Total Evidence Mean: {res['total_evidence_mean']:.2f}")
        print(f"  Max Evidence Mean: {res['max_evidence_mean']:.2f}")

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

    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)

    checkpoint = torch.load('glutton_edl_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    res = evaluate(model, test_loader, criterion, DEVICE, checkpoint['epoch'])

    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Test Loss: {res['loss']:.4f}")
    print(f"Classification Accuracy: {res['accuracy']:.2f}%")
    print(f"Phi MAE: {res['phi_mae']:.2f}°")
    print(f"Psi MAE: {res['psi_mae']:.2f}°")
    print(f"Avg MAE: {(res['phi_mae'] + res['psi_mae'])/2:.2f}°")
    print(f"Uncertainty Mean: {res['uncertainty_mean']:.4f}")
    print(f"Uncertainty-Error Correlation: {res['uncertainty_error_corr']:.3f}")

    # Plot results
    plot_results(res)

    # Plot ensemble examples
    plot_ensemble_example(model, test_loader, DEVICE, n_examples=3, n_samples=100)

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('training_curves_edl.png', dpi=150, bbox_inches='tight')
    print("Training curves saved: training_curves_edl.png")
    plt.close()

    print("\n" + "="*80)
    print("Done! EDL provides uncertainty quantification for ensemble generation.")
    print("High uncertainty -> model unsure -> diverse ensemble")
    print("Low uncertainty -> model confident -> tight ensemble")
    print("="*80)

    return model, res


if __name__ == "__main__":
    model, results = main()