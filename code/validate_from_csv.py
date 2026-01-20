"""
DIRECT VALIDATION FROM CSV
==========================
Your dimensional_stress_data.csv already contains all Phi calculations!
This script validates the theoretical predictions directly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_validate():
    """Load CSV and validate theoretical predictions."""
    
    print("=" * 70)
    print(" DIMENSIONAL LOSS THEOREM VALIDATION")
    print(" Using existing dimensional_stress_data.csv")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('dimensional_stress_data.csv')
    
    print(f"\nLoaded {len(df)} sentences")
    print(f"  Truth: {len(df[df['category']=='truth'])}")
    print(f"  Lies:  {len(df[df['category']=='lie'])}")
    
    # Calculate theoretical predictions for each row
    results = []
    
    for idx, row in df.iterrows():
        N = row['grid_size']
        
        # Theoretical predictions
        R_3d_theory = row['R_2d'] / N
        S_3d_theory = row['S_2d'] * (4/13)  # 8/26 = 4/13
        
        # D_3D theoretical
        p_3d = row['R_2d'] / N
        if p_3d <= 0 or p_3d >= 1:
            D_3d_theory = 0.0
        else:
            D_3d_theory = -p_3d * np.log2(p_3d) - (1-p_3d) * np.log2(1-p_3d)
        
        Phi_3d_theory = R_3d_theory * S_3d_theory + D_3d_theory
        loss_theory = (1 - Phi_3d_theory / row['phi_2d']) * 100 if row['phi_2d'] > 0 else 0
        
        # Calculate errors
        R_error = abs(row['R_3d'] - R_3d_theory) / R_3d_theory * 100 if R_3d_theory > 0 else 0
        S_error = abs(row['S_3d'] - S_3d_theory) / S_3d_theory * 100 if S_3d_theory > 0 else 0
        D_error = abs(row['D_3d'] - D_3d_theory) / D_3d_theory * 100 if D_3d_theory > 0 else 0
        
        results.append({
            'sentence': row['sentence'][:40],
            'category': row['category'],
            'N': N,
            
            # 2D values
            'R_2d': row['R_2d'],
            'S_2d': row['S_2d'],
            'D_2d': row['D_2d'],
            'Phi_2d': row['phi_2d'],
            
            # 3D empirical
            'R_3d_emp': row['R_3d'],
            'S_3d_emp': row['S_3d'],
            'D_3d_emp': row['D_3d'],
            'Phi_3d_emp': row['phi_3d'],
            'loss_emp': row['loss_pct'],
            
            # 3D theoretical
            'R_3d_theory': R_3d_theory,
            'S_3d_theory': S_3d_theory,
            'D_3d_theory': D_3d_theory,
            'Phi_3d_theory': Phi_3d_theory,
            'loss_theory': loss_theory,
            
            # Errors
            'R_error': R_error,
            'S_error': S_error,
            'D_error': D_error,
            'loss_error': abs(row['loss_pct'] - loss_theory)
        })
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Comprehensive analysis."""
    
    print("\n" + "=" * 70)
    print(" VALIDATION RESULTS")
    print("=" * 70)
    
    # Overall statistics
    print(f"\n{'OVERALL STATISTICS':^70}")
    print("-" * 70)
    print(f"Patterns tested: {len(df)}")
    print(f"Grid sizes: {df['N'].min()}-{df['N'].max()} (mean: {df['N'].mean():.1f})")
    print(f"Average density (R_2d): {df['R_2d'].mean():.4f}")
    
    # Information loss
    print(f"\n{'INFORMATION LOSS':^70}")
    print("-" * 70)
    print(f"Empirical loss:    {df['loss_emp'].mean():.2f}% ± {df['loss_emp'].std():.2f}%")
    print(f"Theoretical loss:  {df['loss_theory'].mean():.2f}% ± {df['loss_theory'].std():.2f}%")
    print(f"Discrepancy:       {df['loss_error'].mean():.2f}% ± {df['loss_error'].std():.2f}%")
    
    # Gemini's prediction
    print(f"\nGemini predicted:  86.2% (for p=0.5, N=20)")
    print(f"Your data shows:   {df['loss_emp'].mean():.1f}%")
    print(f"Delta:             {abs(df['loss_emp'].mean() - 86.2):.1f}%")
    
    if abs(df['loss_emp'].mean() - 86.2) < 2:
        print("✓ WITHIN TOLERANCE")
    else:
        print("⚠️ DISCREPANCY DETECTED")
    
    # Component validation
    print(f"\n{'COMPONENT VALIDATION':^70}")
    print("-" * 70)
    
    # S-component (CRITICAL)
    print("\nS-COMPONENT (Connectivity Tax = 18/26):")
    print(f"  Theory: S_3D = (4/13) × S_2D")
    print(f"  Mean error: {df['S_error'].mean():.3f}% ± {df['S_error'].std():.3f}%")
    print(f"  Median error: {df['S_error'].median():.3f}%")
    print(f"  Max error: {df['S_error'].max():.3f}%")
    
    if df['S_error'].mean() < 0.1:
        print("  ✓✓✓ S-COMPONENT EXACT (<0.1% error)")
    elif df['S_error'].mean() < 1:
        print("  ✓✓ S-COMPONENT VALIDATED (<1% error)")
    else:
        print("  ⚠️ S-COMPONENT NEEDS INVESTIGATION")
    
    # R-component
    print("\nR-COMPONENT (Volumetric Dilution):")
    print(f"  Theory: R_3D = R_2D / N")
    print(f"  Mean error: {df['R_error'].mean():.3f}% ± {df['R_error'].std():.3f}%")
    
    if df['R_error'].mean() < 0.01:
        print("  ✓✓✓ R-COMPONENT EXACT")
    
    # D-component
    print("\nD-COMPONENT (Entropy Dilution):")
    print(f"  Theory: D_3D = D(R_2D/N)")
    print(f"  Mean error: {df['D_error'].mean():.3f}% ± {df['D_error'].std():.3f}%")
    
    # Content-blindness
    print(f"\n{'CONTENT-BLINDNESS TEST':^70}")
    print("-" * 70)
    
    truth_loss = df[df['category'] == 'truth']['loss_emp'].mean()
    lie_loss = df[df['category'] == 'lie']['loss_emp'].mean()
    
    t_stat, p_value = stats.ttest_ind(
        df[df['category'] == 'truth']['loss_emp'],
        df[df['category'] == 'lie']['loss_emp']
    )
    
    print(f"Truth loss: {truth_loss:.2f}%")
    print(f"Lie loss:   {lie_loss:.2f}%")
    print(f"Difference: {abs(truth_loss - lie_loss):.2f}%")
    print(f"p-value:    {p_value:.4f}")
    
    if p_value > 0.05:
        print("✓✓✓ CONTENT-BLIND (theorem prediction confirmed)")
    else:
        print("⚠️ Unexpected content-dependence")
    
    # Final verdict
    print("\n" + "=" * 70)
    print(" FINAL VERDICT")
    print("=" * 70)
    
    s_valid = df['S_error'].mean() < 1
    r_valid = df['R_error'].mean() < 0.1
    loss_valid = abs(df['loss_emp'].mean() - 86.2) < 5
    content_blind = p_value > 0.05
    
    if s_valid and r_valid and loss_valid and content_blind:
        print("\n✓✓✓ DIMENSIONAL LOSS THEOREM FULLY VALIDATED")
        print("     All components match theoretical predictions")
        print("     Content-blindness confirmed")
        print("     READY FOR PUBLICATION")
    elif s_valid and r_valid:
        print("\n✓✓ THEOREM COMPONENTS VALIDATED")
        print("    Minor discrepancy in total loss")
        print("    Investigate further before publication")
    else:
        print("\n⚠️ DISCREPANCIES FOUND")
        print("   Formula may need refinement")
    
    return df


def create_visualizations(df):
    """Create validation plots."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # S-component validation
    axes[0, 0].scatter(df['S_2d'], df['S_3d_emp'], alpha=0.6, label='Empirical', s=50)
    axes[0, 0].plot([0, df['S_2d'].max()], [0, df['S_2d'].max()*(4/13)], 
                    'r--', linewidth=2, label='Theory: S×(4/13)')
    axes[0, 0].set_xlabel('S_2D', fontsize=12)
    axes[0, 0].set_ylabel('S_3D', fontsize=12)
    axes[0, 0].set_title('S-Component: 69.2% Loss (18/26)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total Phi
    axes[0, 1].scatter(df['Phi_2d'], df['Phi_3d_emp'], alpha=0.6, label='Empirical', s=50)
    axes[0, 1].plot([0, df['Phi_2d'].max()], [0, df['Phi_2d'].max()/7], 
                    'r--', linewidth=2, label='Theory: Φ/7')
    axes[0, 1].set_xlabel('Φ_2D', fontsize=12)
    axes[0, 1].set_ylabel('Φ_3D', fontsize=12)
    axes[0, 1].set_title('Total Φ: 86% Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1, 0].hist(df['loss_emp'], bins=15, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(df['loss_emp'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f"Mean: {df['loss_emp'].mean():.1f}%")
    axes[1, 0].axvline(86.2, color='green', linestyle='--', 
                      linewidth=2, label='Gemini: 86.2%')
    axes[1, 0].set_xlabel('Information Loss (%)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Content-blindness
    truth_data = df[df['category'] == 'truth']['loss_emp']
    lie_data = df[df['category'] == 'lie']['loss_emp']
    
    bp = axes[1, 1].boxplot([truth_data, lie_data], positions=[1, 2], widths=0.6,
                            patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    axes[1, 1].set_xticklabels(['Truth', 'Lies'], fontsize=12)
    axes[1, 1].set_ylabel('Information Loss (%)', fontsize=12)
    axes[1, 1].set_title('Content-Blindness Test', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(86.2, color='green', linestyle='--', alpha=0.5, label='Theory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add p-value
    t_stat, p_value = stats.ttest_ind(truth_data, lie_data)
    axes[1, 1].text(1.5, axes[1, 1].get_ylim()[1]*0.95, f'p = {p_value:.4f}', 
                   ha='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: validation_results.png")
    
    plt.show()


def main():
    print("\n" + "=" * 70)
    print(" DIRECT CSV VALIDATION")
    print(" No extraction needed - data already computed!")
    print("=" * 70)
    
    # Load and validate
    df = load_and_validate()
    
    # Save detailed results
    df.to_csv('detailed_validation_results.csv', index=False)
    print(f"\n✓ Saved: detailed_validation_results.csv")
    
    # Analyze
    analyze_results(df)
    
    # Visualize
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    print("\n" + "=" * 70)
    print(" VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()