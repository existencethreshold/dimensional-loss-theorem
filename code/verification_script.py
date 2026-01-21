"""
DIMENSIONAL LOSS THEOREM VERIFICATION SCRIPT
============================================

This script validates the theoretical predictions of the Dimensional Loss Theorem
against empirical data from neural attention maps.

Author: Nathan M. Thornhill
Date: January 21, 2026
Purpose: Verify component-wise transformations and total Φ loss predictions

CRITICAL: This script must confirm:
1. S-component loss = 18/26 (geometric theorem)
2. R-component dilution = 1/N (volumetric theorem)
3. D-component entropy change (theoretical formula)
4. Total Φ loss ≈ 86% (combined prediction)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# For loading models (adjust based on your setup)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available - will use saved data if provided")
    MODELS_AVAILABLE = False


# ============================================================================
# CORE PHI CALCULATION FUNCTIONS
# ============================================================================

def calculate_phi_rigorous(grid: np.ndarray, grid_t_minus_1: np.ndarray = None) -> float:
    """
    Calculate Thornhill Φ metric: Φ = R·S + D
    
    This is the CORRECTED version with proper neighbor-sum method for S.
    
    Args:
        grid: Binary grid (2D or 3D)
        grid_t_minus_1: Previous timestep (for dynamic systems). None = static.
    
    Returns:
        Φ value (integrated information measure)
    """
    # Determine dimensionality
    ndim = len(grid.shape)
    
    if ndim == 2:
        k = 8   # Moore neighborhood (2D)
    elif ndim == 3:
        k = 26  # Moore neighborhood (3D)
    else:
        raise ValueError(f"Grid must be 2D or 3D, got {ndim}D")
    
    # ===== R COMPONENT: Information Processing Rate =====
    if grid_t_minus_1 is not None:
        # Dynamic: changed cells / total cells
        changed = np.sum(grid != grid_t_minus_1)
        R = changed / grid.size
    else:
        # Static: active cells / total cells
        active = np.sum(grid)
        R = active / grid.size
    
    if R == 0:
        return 0.0  # Empty grid has Φ = 0
    
    # ===== S COMPONENT: System Integration (Neighbor-Sum Method) =====
    # Find all active cells
    active_coords = np.argwhere(grid == 1)
    
    if len(active_coords) == 0:
        S = 0.0
    else:
        # Count neighbors for each active cell
        neighbor_sum = 0
        
        for coord in active_coords:
            # Get neighbors in Moore neighborhood
            if ndim == 2:
                y, x = coord
                neighbors = grid[max(0, y-1):min(grid.shape[0], y+2),
                               max(0, x-1):min(grid.shape[1], x+2)]
            elif ndim == 3:
                z, y, x = coord
                neighbors = grid[max(0, z-1):min(grid.shape[0], z+2),
                               max(0, y-1):min(grid.shape[1], y+2),
                               max(0, x-1):min(grid.shape[2], x+2)]
            
            # Count active neighbors (excluding self)
            neighbor_count = np.sum(neighbors) - 1  # -1 to exclude center cell
            neighbor_sum += neighbor_count
        
        # S = (total neighbor connections) / (k × active cells)
        S = neighbor_sum / (k * len(active_coords))
    
    # ===== D COMPONENT: Disorder (Binary Shannon Entropy) =====
    if R == 0 or R == 1:
        D = 0.0  # No entropy at extremes
    else:
        D = -R * np.log2(R) - (1-R) * np.log2(1-R)
    
    # ===== TOTAL Φ =====
    Phi = R * S + D
    
    return Phi


def embed_2d_to_3d(pattern_2d: np.ndarray) -> np.ndarray:
    """
    Middle-placement embedding: Place 2D pattern at middle z-slice of 3D cube.
    
    This is the proven method that causes 86% information loss.
    
    Args:
        pattern_2d: Binary 2D array (N×N)
    
    Returns:
        Binary 3D array (N×N×N) with pattern at z = N//2
    """
    N = pattern_2d.shape[0]
    cube_3d = np.zeros((N, N, N), dtype=int)
    middle_z = N // 2
    cube_3d[:, :, middle_z] = pattern_2d
    return cube_3d


def calculate_components_detailed(grid: np.ndarray) -> Dict[str, float]:
    """
    Calculate all components of Φ separately for analysis.
    
    Returns dict with R, S, D, and Φ values.
    """
    ndim = len(grid.shape)
    k = 8 if ndim == 2 else 26
    
    # R component
    R = np.sum(grid) / grid.size
    
    # S component
    active_coords = np.argwhere(grid == 1)
    if len(active_coords) == 0:
        S = 0.0
    else:
        neighbor_sum = 0
        for coord in active_coords:
            if ndim == 2:
                y, x = coord
                neighbors = grid[max(0, y-1):min(grid.shape[0], y+2),
                               max(0, x-1):min(grid.shape[1], x+2)]
            elif ndim == 3:
                z, y, x = coord
                neighbors = grid[max(0, z-1):min(grid.shape[0], z+2),
                               max(0, y-1):min(grid.shape[1], y+2),
                               max(0, x-1):min(grid.shape[2], x+2)]
            neighbor_count = np.sum(neighbors) - 1
            neighbor_sum += neighbor_count
        S = neighbor_sum / (k * len(active_coords))
    
    # D component
    if R == 0 or R == 1:
        D = 0.0
    else:
        D = -R * np.log2(R) - (1-R) * np.log2(1-R)
    
    # Total Φ
    Phi = R * S + D
    
    return {
        'R': R,
        'S': S,
        'D': D,
        'Phi': Phi,
        'R_times_S': R * S,
        'k': k
    }


# ============================================================================
# ATTENTION MAP EXTRACTION
# ============================================================================

def get_attention_2d_from_model(text: str, model_name: str = "gpt2") -> np.ndarray:
    """
    Extract 2D attention map from language model.
    
    Args:
        text: Input sentence
        model_name: HuggingFace model name
    
    Returns:
        2D attention matrix (averaged over heads)
    """
    if not MODELS_AVAILABLE:
        raise RuntimeError("Transformers library not available")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get attention
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last layer attention, average over heads
    attention = outputs.attentions[-1]  # Last layer
    attention_avg = attention.mean(dim=1).squeeze().numpy()  # Average heads
    
    return attention_avg


def binarize_attention(attention: np.ndarray, percentile: float = 90) -> np.ndarray:
    """
    Binarize attention map at given percentile threshold.
    
    Args:
        attention: 2D attention matrix
        percentile: Threshold percentile (default 90 = top 10%)
    
    Returns:
        Binary 2D array
    """
    threshold = np.percentile(attention, percentile)
    return (attention > threshold).astype(int)


# ============================================================================
# THEORETICAL PREDICTIONS
# ============================================================================

def predict_components_3d(R_2d: float, S_2d: float, D_2d: float, N: int) -> Dict[str, float]:
    """
    Calculate theoretical predictions for 3D components.
    
    Based on Dimensional Loss Theorem:
    - R_3D = R_2D / N (volumetric dilution)
    - S_3D = (4/13) × S_2D (connectivity tax)
    - D_3D = D(R_2D/N) (entropy formula)
    
    Args:
        R_2d, S_2d, D_2d: 2D components
        N: Grid size
    
    Returns:
        Dict with predicted 3D components
    """
    # R transformation (volumetric dilution)
    R_3d_theory = R_2d / N
    
    # S transformation (connectivity tax: 8 accessible out of 26 total)
    S_3d_theory = S_2d * (4/13)
    
    # D transformation (entropy of diluted density)
    p_3d = R_2d / N
    if p_3d == 0 or p_3d >= 1:
        D_3d_theory = 0.0
    else:
        D_3d_theory = -p_3d * np.log2(p_3d) - (1-p_3d) * np.log2(1-p_3d)
    
    # Predicted Φ_3D
    Phi_3d_theory = R_3d_theory * S_3d_theory + D_3d_theory
    
    return {
        'R_3d_theory': R_3d_theory,
        'S_3d_theory': S_3d_theory,
        'D_3d_theory': D_3d_theory,
        'Phi_3d_theory': Phi_3d_theory,
        'RS_3d_theory': R_3d_theory * S_3d_theory
    }


# ============================================================================
# MAIN VALIDATION FUNCTIONS
# ============================================================================

def validate_single_pattern(pattern_2d: np.ndarray, label: str = "") -> Dict:
    """
    Validate dimensional loss theorem on a single 2D pattern.
    
    Args:
        pattern_2d: Binary 2D array
        label: Description of pattern
    
    Returns:
        Dict with all measurements and comparisons
    """
    N = pattern_2d.shape[0]
    
    # Calculate 2D components
    comp_2d = calculate_components_detailed(pattern_2d)
    
    # Embed to 3D
    pattern_3d = embed_2d_to_3d(pattern_2d)
    
    # Calculate 3D components
    comp_3d = calculate_components_detailed(pattern_3d)
    
    # Theoretical predictions
    theory = predict_components_3d(comp_2d['R'], comp_2d['S'], comp_2d['D'], N)
    
    # Calculate errors
    R_error = abs(comp_3d['R'] - theory['R_3d_theory']) / theory['R_3d_theory'] * 100 if theory['R_3d_theory'] > 0 else 0
    S_error = abs(comp_3d['S'] - theory['S_3d_theory']) / theory['S_3d_theory'] * 100 if theory['S_3d_theory'] > 0 else 0
    D_error = abs(comp_3d['D'] - theory['D_3d_theory']) / theory['D_3d_theory'] * 100 if theory['D_3d_theory'] > 0 else 0
    
    # Total loss
    loss_pct = (1 - comp_3d['Phi'] / comp_2d['Phi']) * 100 if comp_2d['Phi'] > 0 else 0
    loss_theory_pct = (1 - theory['Phi_3d_theory'] / comp_2d['Phi']) * 100 if comp_2d['Phi'] > 0 else 0
    
    return {
        'label': label,
        'N': N,
        
        # 2D components
        'R_2d': comp_2d['R'],
        'S_2d': comp_2d['S'],
        'D_2d': comp_2d['D'],
        'Phi_2d': comp_2d['Phi'],
        'RS_2d': comp_2d['R_times_S'],
        
        # 3D components (empirical)
        'R_3d_empirical': comp_3d['R'],
        'S_3d_empirical': comp_3d['S'],
        'D_3d_empirical': comp_3d['D'],
        'Phi_3d_empirical': comp_3d['Phi'],
        'RS_3d_empirical': comp_3d['R_times_S'],
        
        # 3D components (theoretical)
        'R_3d_theory': theory['R_3d_theory'],
        'S_3d_theory': theory['S_3d_theory'],
        'D_3d_theory': theory['D_3d_theory'],
        'Phi_3d_theory': theory['Phi_3d_theory'],
        'RS_3d_theory': theory['RS_3d_theory'],
        
        # Errors
        'R_error_pct': R_error,
        'S_error_pct': S_error,
        'D_error_pct': D_error,
        
        # Loss
        'loss_pct_empirical': loss_pct,
        'loss_pct_theory': loss_theory_pct,
        'loss_error': abs(loss_pct - loss_theory_pct)
    }


def validate_dataset(sentences: List[str], model_name: str = "gpt2") -> pd.DataFrame:
    """
    Validate theorem across entire dataset of sentences.
    
    Args:
        sentences: List of text sentences
        model_name: Model to extract attention from
    
    Returns:
        DataFrame with all results
    """
    results = []
    
    print(f"Validating {len(sentences)} sentences with {model_name}...")
    print("=" * 70)
    
    for i, sentence in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}] {sentence[:50]}...")
        
        try:
            # Get attention map
            attention = get_attention_2d_from_model(sentence, model_name)
            
            # Binarize at 90th percentile
            pattern_2d = binarize_attention(attention, percentile=90)
            
            # Validate
            result = validate_single_pattern(pattern_2d, label=sentence[:30])
            results.append(result)
            
            # Print key results
            print(f"  N={result['N']}, Φ_2D={result['Phi_2d']:.4f}, Φ_3D={result['Phi_3d_empirical']:.4f}")
            print(f"  Loss: {result['loss_pct_empirical']:.1f}% (empirical) vs {result['loss_pct_theory']:.1f}% (theory)")
            print(f"  S-error: {result['S_error_pct']:.2f}%")
            
        except Exception as e:
            print(f"  ⚠️ ERROR: {e}")
            continue
    
    return pd.DataFrame(results)


def validate_from_saved_data(data_path: str) -> pd.DataFrame:
    """
    Validate theorem using pre-extracted attention maps.
    
    Useful if you've already saved your attention maps to avoid re-running models.
    
    Args:
        data_path: Path to .npy file containing attention maps
                   Format: dict with keys 'truths' and 'lies', values are lists of 2D arrays
    
    Returns:
        DataFrame with results
    """
    import os
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"\n"
            f"To generate this data, you need to:\n"
            f"1. Install transformers: pip install transformers torch\n"
            f"2. Set USE_SAVED_DATA = False in the main() function\n"
            f"\n"
            f"Or run the CSV validation instead: python validate_from_csv.py"
        )
    
    print(f"Loading saved data from {data_path}...")
    data = np.load(data_path, allow_pickle=True).item()
    
    results = []
    
    for category in ['truths', 'lies']:
        if category not in data:
            continue
            
        patterns = data[category]
        print(f"\nProcessing {len(patterns)} {category}...")
        
        for i, pattern_2d in enumerate(patterns):
            label = f"{category}_{i}"
            result = validate_single_pattern(pattern_2d, label=label)
            result['category'] = category
            results.append(result)
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(patterns)}")
    
    return pd.DataFrame(results)


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_results(df: pd.DataFrame) -> None:
    """
    Comprehensive statistical analysis of validation results.
    
    Prints detailed analysis to console.
    """
    print("\n" + "=" * 70)
    print(" DIMENSIONAL LOSS THEOREM VALIDATION RESULTS")
    print("=" * 70)
    
    # Overall statistics
    print(f"\n{'OVERALL STATISTICS':^70}")
    print("-" * 70)
    print(f"Total patterns tested: {len(df)}")
    print(f"Grid sizes: {df['N'].min()}-{df['N'].max()} (mean: {df['N'].mean():.1f})")
    print(f"Average density (R_2d): {df['R_2d'].mean():.4f} ± {df['R_2d'].std():.4f}")
    
    # Information loss
    print(f"\n{'INFORMATION LOSS':^70}")
    print("-" * 70)
    print(f"Empirical loss:    {df['loss_pct_empirical'].mean():.2f}% ± {df['loss_pct_empirical'].std():.2f}%")
    print(f"Theoretical loss:  {df['loss_pct_theory'].mean():.2f}% ± {df['loss_pct_theory'].std():.2f}%")
    print(f"Discrepancy:       {df['loss_error'].mean():.2f}% ± {df['loss_error'].std():.2f}%")
    
    # Expected value from theoretical calculation
    expected_loss = 86.2  # For p=0.5, N=20 (theoretical baseline)
    actual_mean_loss = df['loss_pct_empirical'].mean()
    print(f"\nTheoretical (p=0.5, N=20): {expected_loss:.1f}%")
    print(f"Observed (this dataset):    {actual_mean_loss:.1f}%")
    print(f"Delta:                      {abs(actual_mean_loss - expected_loss):.1f}%")
    
    if abs(actual_mean_loss - expected_loss) < 2:
        print("✓ WITHIN TOLERANCE (< 2%)")
    else:
        print("⚠️ EXCEEDS TOLERANCE - INVESTIGATE")
    
    # Component validation
    print(f"\n{'COMPONENT VALIDATION':^70}")
    print("-" * 70)
    
    # S-component (most critical - should be EXACT)
    print("\nS-COMPONENT (Connectivity Tax):")
    print(f"  Predicted: S_3D = (4/13) × S_2D")
    print(f"  Mean error: {df['S_error_pct'].mean():.2f}% ± {df['S_error_pct'].std():.2f}%")
    print(f"  Median error: {df['S_error_pct'].median():.2f}%")
    print(f"  Max error: {df['S_error_pct'].max():.2f}%")
    
    if df['S_error_pct'].mean() < 1:
        print("  ✓ S-COMPONENT THEOREM VALIDATED (< 1% error)")
    elif df['S_error_pct'].mean() < 5:
        print("  ⚠️ S-COMPONENT MOSTLY VALIDATED (< 5% error)")
    else:
        print("  ✗ S-COMPONENT THEOREM FAILED (> 5% error)")
    
    # R-component
    print("\nR-COMPONENT (Volumetric Dilution):")
    print(f"  Predicted: R_3D = R_2D / N")
    print(f"  Mean error: {df['R_error_pct'].mean():.2f}% ± {df['R_error_pct'].std():.2f}%")
    
    if df['R_error_pct'].mean() < 0.1:
        print("  ✓ R-COMPONENT EXACT (< 0.1% error)")
    
    # D-component
    print("\nD-COMPONENT (Entropy Dilution):")
    print(f"  Predicted: D_3D = D(R_2D/N)")
    print(f"  Mean error: {df['D_error_pct'].mean():.2f}% ± {df['D_error_pct'].std():.2f}%")
    
    # RS negligibility
    print(f"\n{'R×S NEGLIGIBILITY':^70}")
    print("-" * 70)
    rs_2d_mean = df['RS_2d'].mean()
    rs_3d_mean = df['RS_3d_empirical'].mean()
    rs_collapse = (1 - rs_3d_mean / rs_2d_mean) * 100 if rs_2d_mean > 0 else 0
    
    print(f"R×S in 2D: {rs_2d_mean:.6f}")
    print(f"R×S in 3D: {rs_3d_mean:.6f}")
    print(f"Collapse:  {rs_collapse:.2f}%")
    
    if rs_collapse > 95:
        print("✓ R×S BECOMES NEGLIGIBLE (>95% collapse)")
    
    # Content-blindness (if categories available)
    if 'category' in df.columns:
        print(f"\n{'CONTENT-BLINDNESS TEST':^70}")
        print("-" * 70)
        
        truth_loss = df[df['category'] == 'truths']['loss_pct_empirical'].mean()
        lie_loss = df[df['category'] == 'lies']['loss_pct_empirical'].mean()
        
        t_stat, p_value = stats.ttest_ind(
            df[df['category'] == 'truths']['loss_pct_empirical'],
            df[df['category'] == 'lies']['loss_pct_empirical']
        )
        
        print(f"Truth loss: {truth_loss:.2f}%")
        print(f"Lie loss:   {lie_loss:.2f}%")
        print(f"Difference: {abs(truth_loss - lie_loss):.2f}%")
        print(f"p-value:    {p_value:.4f}")
        
        if p_value > 0.05:
            print("✓ CONTENT-BLIND (p > 0.05) - Theorem prediction confirmed")
        else:
            print("⚠️ Content-dependent (p < 0.05) - Unexpected")


def create_visualizations(df: pd.DataFrame, output_dir: str = ".") -> None:
    """
    Create comprehensive visualization suite.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save figures
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Component transformations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R transformation
    axes[0, 0].scatter(df['R_2d'], df['R_3d_empirical'], alpha=0.6, label='Empirical')
    axes[0, 0].scatter(df['R_2d'], df['R_3d_theory'], alpha=0.6, marker='x', label='Theory')
    axes[0, 0].plot([0, df['R_2d'].max()], [0, df['R_2d'].max()/df['N'].mean()], 
                    'r--', alpha=0.5, label=f'R/N (N≈{df["N"].mean():.0f})')
    axes[0, 0].set_xlabel('R_2D')
    axes[0, 0].set_ylabel('R_3D')
    axes[0, 0].set_title('R-Component: Volumetric Dilution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # S transformation
    axes[0, 1].scatter(df['S_2d'], df['S_3d_empirical'], alpha=0.6, label='Empirical')
    axes[0, 1].scatter(df['S_2d'], df['S_3d_theory'], alpha=0.6, marker='x', label='Theory')
    axes[0, 1].plot([0, df['S_2d'].max()], [0, df['S_2d'].max()*(4/13)], 
                    'r--', alpha=0.5, label='S×(4/13)')
    axes[0, 1].set_xlabel('S_2D')
    axes[0, 1].set_ylabel('S_3D')
    axes[0, 1].set_title('S-Component: Connectivity Tax (18/26)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # D transformation
    axes[1, 0].scatter(df['D_2d'], df['D_3d_empirical'], alpha=0.6, label='Empirical')
    axes[1, 0].scatter(df['D_2d'], df['D_3d_theory'], alpha=0.6, marker='x', label='Theory')
    axes[1, 0].set_xlabel('D_2D')
    axes[1, 0].set_ylabel('D_3D')
    axes[1, 0].set_title('D-Component: Entropy Dilution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total Φ
    axes[1, 1].scatter(df['Phi_2d'], df['Phi_3d_empirical'], alpha=0.6, label='Empirical')
    axes[1, 1].scatter(df['Phi_2d'], df['Phi_3d_theory'], alpha=0.6, marker='x', label='Theory')
    axes[1, 1].plot([0, df['Phi_2d'].max()], [0, df['Phi_2d'].max()/7], 
                    'r--', alpha=0.5, label='Φ/7 (1/7 retention)')
    axes[1, 1].set_xlabel('Φ_2D')
    axes[1, 1].set_ylabel('Φ_3D')
    axes[1, 1].set_title('Total Φ: Combined Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_transformations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/component_transformations.png")
    
    # Figure 2: Loss distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['loss_pct_empirical'], bins=20, alpha=0.7, edgecolor='black', label='Empirical')
    axes[0].axvline(df['loss_pct_empirical'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df["loss_pct_empirical"].mean():.1f}%')
    axes[0].axvline(86.2, color='green', linestyle='--', 
                    linewidth=2, label='Theoretical: 86.2%')
    axes[0].set_xlabel('Information Loss (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Φ Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1].hist(df['S_error_pct'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(df['S_error_pct'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df["S_error_pct"].mean():.2f}%')
    axes[1].set_xlabel('S-Component Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('S-Component Prediction Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/loss_distributions.png")
    
    # Figure 3: Content-blindness (if applicable)
    if 'category' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        truth_data = df[df['category'] == 'truths']['loss_pct_empirical']
        lie_data = df[df['category'] == 'lies']['loss_pct_empirical']
        
        positions = [1, 2]
        box_data = [truth_data, lie_data]
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax.set_xticklabels(['Truth', 'Lies'])
        ax.set_ylabel('Information Loss (%)')
        ax.set_title('Content-Blindness: Truth vs Lies')
        ax.axhline(86.2, color='green', linestyle='--', alpha=0.5, label='Theoretical')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add p-value
        t_stat, p_value = stats.ttest_ind(truth_data, lie_data)
        ax.text(1.5, ax.get_ylim()[1]*0.95, f'p = {p_value:.4f}', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/content_blindness.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/content_blindness.png")
    
    plt.close('all')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main validation script.
    
    Run this to validate the Dimensional Loss Theorem.
    """
    print("\n" + "=" * 70)
    print(" DIMENSIONAL LOSS THEOREM VALIDATION")
    print(" Nathan M. Thornhill - January 21, 2026")
    print("=" * 70)
    
    # ===== IMPORTANT: READ THIS FIRST =====
    # This script requires either:
    # 1. Pre-saved attention_maps.npy file (advanced users)
    # 2. OR transformers library installed (downloads ~500MB)
    #
    # RECOMMENDED FOR MOST USERS: Use validate_from_csv.py instead!
    # That script uses pre-computed data and works immediately.
    #
    # Only use THIS script if you want to test on actual neural networks.
    
    # ===== CONFIGURATION =====
    
    # Option 1: Load from saved data (requires attention_maps.npy file)
    USE_SAVED_DATA = True
    SAVED_DATA_PATH = "attention_maps.npy"
    
    # Option 2: Extract from models (downloads ~500MB of models on first run)
    # Set USE_SAVED_DATA = False to use this option
    MODEL_NAME = "gpt2"
    
    OUTPUT_DIR = "validation_results"
    
    # Create output directory
    import os
    import sys
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try to load sentences from data/test_sentences.py
    try:
        # Multiple path attempts for robustness
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, '..', 'data'),
            os.path.join(script_dir, '..', '..', 'data'),  # For nested structures
            os.path.join(os.getcwd(), 'data'),
            os.path.join(os.getcwd(), '..', 'data'),
        ]
        
        loaded = False
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'test_sentences.py')):
                sys.path.insert(0, path)
                from test_sentences import coherent, incoherent
                SENTENCES = coherent + incoherent
                print(f"✓ Loaded {len(SENTENCES)} test sentences from {path}")
                loaded = True
                break
        
        if not loaded:
            raise ImportError("Could not find test_sentences.py")
            
    except Exception as e:
        # Fallback to example sentences
        print(f"⚠️  Could not load test_sentences.py: {e}")
        SENTENCES = [
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower is located on the moon.",
            # Note: For full validation, you need the complete test_sentences.py file
        ]
        print(f"⚠️  Using only {len(SENTENCES)} example sentences - results NOT representative!")
    
    # ===== RUN VALIDATION =====
    if USE_SAVED_DATA:
        print(f"\nLooking for saved data: {SAVED_DATA_PATH}")
        if not os.path.exists(SAVED_DATA_PATH):
            # Check if we can fall back to model extraction
            if MODELS_AVAILABLE:
                print(f"⚠️  Saved data not found: {SAVED_DATA_PATH}")
                print(f"\n✓ Transformers library detected!")
                print(f"\nSwitching to model extraction mode...")
                print(f"This will download ~500MB of models on first run.")
                print(f"\nPress Ctrl+C now to cancel, or wait 5 seconds to continue...")
                import time
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    print("\n\nCancelled by user.")
                    return
                
                print(f"\nExtracting attention from {len(SENTENCES)} sentences using {MODEL_NAME}...")
                df = validate_dataset(SENTENCES, model_name=MODEL_NAME)
            else:
                print(f"\n⚠️  ERROR: Saved data not found and transformers not available.")
                print(f"\nTo use this script, you need EITHER:")
                print(f"  1. Pre-saved attention_maps.npy file (not included in repo)")
                print(f"  2. Transformers library: pip install transformers torch")
                print(f"\nRECOMMENDED: Use validate_from_csv.py instead!")
                print(f"  It uses pre-computed data and works immediately.")
                return
        else:
            print(f"Loading saved data from {SAVED_DATA_PATH}...")
            df = validate_from_saved_data(SAVED_DATA_PATH)
    else:
        print(f"\nExtracting attention from {len(SENTENCES)} sentences using {MODEL_NAME}")
        df = validate_dataset(SENTENCES, model_name=MODEL_NAME)
    
    # Save raw results
    df.to_csv(f'{OUTPUT_DIR}/validation_results.csv', index=False)
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/validation_results.csv")
    
    # ===== ANALYSIS =====
    analyze_results(df)
    
    # ===== VISUALIZATIONS =====
    print(f"\nCreating visualizations...")
    create_visualizations(df, output_dir=OUTPUT_DIR)
    
    # ===== FINAL VERDICT =====
    print("\n" + "=" * 70)
    print(" FINAL VERDICT")
    print("=" * 70)
    
    s_error_mean = df['S_error_pct'].mean()
    loss_mean = df['loss_pct_empirical'].mean()
    loss_std = df['loss_pct_empirical'].std()
    
    print(f"\nS-Component Error: {s_error_mean:.2f}%")
    if s_error_mean < 1:
        print("✓✓✓ S-COMPONENT THEOREM RIGOROUSLY VALIDATED")
    elif s_error_mean < 5:
        print("✓✓  S-COMPONENT MOSTLY VALIDATED")
    else:
        print("✗   S-COMPONENT NEEDS REVISION")
    
    print(f"\nObserved Loss: {loss_mean:.2f}% ± {loss_std:.2f}%")
    print(f"Theoretical Baseline: 84-86%")
    print(f"Discrepancy:   {abs(loss_mean - 86.2):.2f}%")
    
    if abs(loss_mean - 86.2) < 2:
        print("\n✓✓✓ DIMENSIONAL LOSS THEOREM VALIDATED")
        print("     Proceed with publication.")
    elif abs(loss_mean - 86.2) < 5:
        print("\n✓✓  THEOREM MOSTLY VALIDATED")
        print("     Minor refinements needed.")
    else:
        print("\n✗   SIGNIFICANT DISCREPANCY")
        print("     Formula requires revision.")
    
    print("\n" + "=" * 70)
    print(f" Analysis complete. Check {OUTPUT_DIR}/ for full results.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()