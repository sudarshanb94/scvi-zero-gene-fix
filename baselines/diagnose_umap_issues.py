#!/usr/bin/env python3
"""
Comprehensive diagnostic to identify why real and reconstructed UMAPs don't overlap.
"""

import sys
from pathlib import Path
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml


def diagnose_reconstruction_quality(model, dataloader, device='cuda', max_batches=10):
    """Comprehensive diagnostic of reconstruction quality."""
    model.eval()
    model = model.to(device)
    
    all_real = []
    all_recon = []
    all_real_counts = []
    all_recon_counts = []
    
    print("=" * 80)
    print("COMPREHENSIVE RECONSTRUCTION DIAGNOSTIC")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Use predict_step to get normalized data (for UMAP comparison)
            batch_preds = model.predict_step(batch, batch_idx)
            x_real_log = batch_preds["X"].cpu().numpy()
            x_recon_log = batch_preds["preds"].cpu().numpy()
            
            # Also get raw counts for analysis
            x_pert, x_basal, pert, cell_type, batch_ids = model.extract_batch_tensors(batch)
            if x_pert.max() <= 25.0:
                x_real_counts = (torch.exp(x_pert) - 1).cpu().numpy()
            else:
                x_real_counts = x_pert.cpu().numpy()
            
            # Get raw reconstruction
            if x_basal.max() <= 25.0:
                x_basal_counts = torch.exp(x_basal) - 1
            else:
                x_basal_counts = x_basal
            
            encoder_outputs, decoder_outputs = model.module.forward(
                x_basal_counts, pert, cell_type, batch_ids
            )
            if model.recon_loss == "gauss":
                x_recon_counts = decoder_outputs["px"].loc.cpu().numpy()
            else:
                x_recon_counts = decoder_outputs["px"].mu.cpu().numpy()
            
            all_real.append(x_real_log)
            all_recon.append(x_recon_log)
            all_real_counts.append(x_real_counts)
            all_recon_counts.append(x_recon_counts)
    
    # Concatenate
    real_log = np.vstack(all_real)
    recon_log = np.vstack(all_recon)
    real_counts = np.vstack(all_real_counts)
    recon_counts = np.vstack(all_recon_counts)
    
    print(f"\n1. DATA SHAPE AND RANGES")
    print("-" * 80)
    print(f"Real (log-normalized): shape={real_log.shape}, range=[{real_log.min():.2f}, {real_log.max():.2f}], mean={real_log.mean():.2f}")
    print(f"Recon (log-normalized): shape={recon_log.shape}, range=[{recon_log.min():.2f}, {recon_log.max():.2f}], mean={recon_log.mean():.2f}")
    print(f"Real (counts): shape={real_counts.shape}, range=[{real_counts.min():.2f}, {real_counts.max():.2f}], mean={real_counts.mean():.2f}")
    print(f"Recon (counts): shape={recon_counts.shape}, range=[{recon_counts.min():.2f}, {recon_counts.max():.2f}], mean={recon_counts.mean():.2f}")
    
    print(f"\n2. GLOBAL RECONSTRUCTION METRICS (log-normalized space)")
    print("-" * 80)
    mse = mean_squared_error(real_log.flatten(), recon_log.flatten())
    r2 = r2_score(real_log.flatten(), recon_log.flatten())
    corr = np.corrcoef(real_log.flatten(), recon_log.flatten())[0, 1]
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson correlation: {corr:.4f}")
    
    print(f"\n3. PER-CELL RECONSTRUCTION QUALITY")
    print("-" * 80)
    cell_mse = np.mean((real_log - recon_log) ** 2, axis=1)
    cell_corr = [np.corrcoef(real_log[i], recon_log[i])[0, 1] for i in range(len(real_log))]
    print(f"Per-cell MSE: mean={cell_mse.mean():.4f}, std={cell_mse.std():.4f}, min={cell_mse.min():.4f}, max={cell_mse.max():.4f}")
    print(f"Per-cell correlation: mean={np.mean(cell_corr):.4f}, std={np.std(cell_corr):.4f}, min={np.min(cell_corr):.4f}, max={np.max(cell_corr):.4f}")
    print(f"Cells with correlation < 0.5: {(np.array(cell_corr) < 0.5).sum()} / {len(cell_corr)} ({(np.array(cell_corr) < 0.5).sum() / len(cell_corr) * 100:.1f}%)")
    
    print(f"\n4. PER-GENE RECONSTRUCTION QUALITY")
    print("-" * 80)
    gene_mse = np.mean((real_log - recon_log) ** 2, axis=0)
    gene_corr = [np.corrcoef(real_log[:, i], recon_log[:, i])[0, 1] for i in range(real_log.shape[1])]
    print(f"Per-gene MSE: mean={gene_mse.mean():.4f}, std={gene_mse.std():.4f}, min={gene_mse.min():.4f}, max={gene_mse.max():.4f}")
    print(f"Per-gene correlation: mean={np.mean(gene_corr):.4f}, std={np.std(gene_corr):.4f}, min={np.min(gene_corr):.4f}, max={np.max(gene_corr):.4f}")
    print(f"Genes with correlation < 0.3: {(np.array(gene_corr) < 0.3).sum()} / {len(gene_corr)} ({(np.array(gene_corr) < 0.3).sum() / len(gene_corr) * 100:.1f}%)")
    
    print(f"\n5. ZERO GENE ANALYSIS")
    print("-" * 80)
    # Find always-zero genes
    gene_sums = real_counts.sum(axis=0)
    always_zero_genes = gene_sums == 0
    sometimes_zero_genes = (real_counts == 0).any(axis=0) & ~always_zero_genes
    never_zero_genes = (real_counts > 0).all(axis=0)
    
    print(f"Always-zero genes: {always_zero_genes.sum()} ({always_zero_genes.sum() / len(gene_sums) * 100:.1f}%)")
    print(f"Sometimes-zero genes: {sometimes_zero_genes.sum()} ({sometimes_zero_genes.sum() / len(gene_sums) * 100:.1f}%)")
    print(f"Never-zero genes: {never_zero_genes.sum()} ({never_zero_genes.sum() / len(gene_sums) * 100:.1f}%)")
    
    if always_zero_genes.sum() > 0:
        always_zero_recon = recon_counts[:, always_zero_genes]
        print(f"\n  Always-zero genes reconstruction:")
        print(f"    Mean prediction: {always_zero_recon.mean():.4f}")
        print(f"    Max prediction: {always_zero_recon.max():.4f}")
        print(f"    Genes with mean > 0.01: {(always_zero_recon.mean(axis=0) > 0.01).sum()}")
        print(f"    ⚠️  This could be polluting the reconstruction space!")
    
    print(f"\n6. DISTRIBUTION COMPARISON")
    print("-" * 80)
    # Compare distributions
    real_flat = real_log.flatten()
    recon_flat = recon_log.flatten()
    
    # Remove zeros for better comparison
    real_nonzero = real_flat[real_flat > 0]
    recon_nonzero = recon_flat[recon_flat > 0]
    
    print(f"Real data: {len(real_nonzero)} non-zero values, mean={real_nonzero.mean():.4f}, std={real_nonzero.std():.4f}")
    print(f"Recon data: {len(recon_nonzero)} non-zero values, mean={recon_nonzero.mean():.4f}, std={recon_nonzero.std():.4f}")
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(real_flat, recon_flat)
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    if ks_p < 0.05:
        print(f"  ⚠️  Distributions are significantly different!")
    
    print(f"\n7. PCA ANALYSIS (for UMAP)")
    print("-" * 80)
    # Check PCA on joint space
    joint_data = np.vstack([real_log, recon_log])
    pca = PCA(n_components=50, random_state=42)
    joint_pca = pca.fit_transform(joint_data)
    
    print(f"Joint PCA (50 components): explained variance = {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  First 10 components: {pca.explained_variance_ratio_[:10]}")
    
    # Check if real and recon separate in PCA space
    real_pca = joint_pca[:len(real_log)]
    recon_pca = joint_pca[len(real_log):]
    
    # Distance between centroids
    real_centroid = real_pca.mean(axis=0)
    recon_centroid = recon_pca.mean(axis=0)
    centroid_dist = np.linalg.norm(real_centroid - recon_centroid)
    print(f"  Distance between real and recon centroids in PCA space: {centroid_dist:.4f}")
    
    # Check variance in each group
    real_pca_var = real_pca.var(axis=0).sum()
    recon_pca_var = recon_pca.var(axis=0).sum()
    print(f"  Real data variance in PCA space: {real_pca_var:.4f}")
    print(f"  Recon data variance in PCA space: {recon_pca_var:.4f}")
    print(f"  Variance ratio (recon/real): {recon_pca_var / real_pca_var:.4f}")
    if recon_pca_var / real_pca_var > 1.5 or recon_pca_var / real_pca_var < 0.67:
        print(f"  ⚠️  Significant variance difference - could cause UMAP misalignment!")
    
    print(f"\n8. SPARSE VS DENSE GENE ANALYSIS")
    print("-" * 80)
    # Check if sparse genes are handled differently
    gene_sparsity = (real_counts == 0).mean(axis=0)  # Fraction of zeros per gene
    
    # Bin genes by sparsity
    sparse_genes = gene_sparsity > 0.9  # >90% zeros
    medium_genes = (gene_sparsity > 0.1) & (gene_sparsity <= 0.9)
    dense_genes = gene_sparsity <= 0.1
    
    print(f"Sparse genes (>90% zeros): {sparse_genes.sum()}")
    print(f"Medium genes (10-90% zeros): {medium_genes.sum()}")
    print(f"Dense genes (<10% zeros): {dense_genes.sum()}")
    
    if sparse_genes.sum() > 0:
        sparse_corr = np.array(gene_corr)[sparse_genes]
        print(f"  Sparse gene correlation: mean={sparse_corr.mean():.4f}")
    
    if dense_genes.sum() > 0:
        dense_corr = np.array(gene_corr)[dense_genes]
        print(f"  Dense gene correlation: mean={dense_corr.mean():.4f}")
    
    if sparse_genes.sum() > 0 and dense_genes.sum() > 0:
        if sparse_corr.mean() < dense_corr.mean() - 0.2:
            print(f"  ⚠️  Sparse genes are poorly reconstructed - this could affect UMAP!")
    
    print(f"\n9. RECOMMENDATIONS")
    print("-" * 80)
    issues = []
    
    if always_zero_genes.sum() > 0 and (recon_counts[:, always_zero_genes].mean(axis=0) > 0.01).sum() > 10:
        issues.append("Always-zero genes have high predictions - increase zero-gene penalty weight")
    
    if np.mean(cell_corr) < 0.5:
        issues.append(f"Low per-cell correlation ({np.mean(cell_corr):.4f}) - model may need more training")
    
    if np.mean(gene_corr) < 0.4:
        issues.append(f"Low per-gene correlation ({np.mean(gene_corr):.4f}) - reconstruction quality is poor")
    
    if recon_pca_var / real_pca_var > 1.5 or recon_pca_var / real_pca_var < 0.67:
        issues.append("Variance mismatch in PCA space - could cause UMAP misalignment")
    
    if sparse_genes.sum() > 0 and dense_genes.sum() > 0:
        if np.array(gene_corr)[sparse_genes].mean() < np.array(gene_corr)[dense_genes].mean() - 0.2:
            issues.append("Sparse genes poorly reconstructed - consider gene weighting in loss")
    
    if len(issues) == 0:
        print("✓ No major issues detected. UMAP misalignment might be due to:")
        print("  - UMAP hyperparameters (n_neighbors, min_dist)")
        print("  - Need for more training")
        print("  - Inherent differences in reconstruction vs real data")
    else:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    print("\n" + "=" * 80)
    
    return {
        'real_log': real_log,
        'recon_log': recon_log,
        'real_counts': real_counts,
        'recon_counts': recon_counts,
        'cell_corr': cell_corr,
        'gene_corr': gene_corr,
        'always_zero_genes': always_zero_genes,
        'issues': issues
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose UMAP mapping issues')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_module', type=str, required=True,
                       help='Path to data module (or config will be used)')
    parser.add_argument('--max_batches', type=int, default=10,
                       help='Number of batches to analyze')
    
    args = parser.parse_args()
    
    # Load data module
    if Path(args.data_module).exists():
        with open(args.data_module, 'rb') as f:
            datamodule = pickle.load(f)
    else:
        config_path = Path(args.checkpoint).parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_config = OmegaConf.create(config['data'])
        datamodule = get_datamodule(
            name=data_config['name'],
            kwargs=data_config['kwargs'],
            batch_size=data_config['kwargs'].get('batch_size', 128),
            cell_sentence_len=data_config['kwargs'].get('cell_sentence_len', 1)
        )
        datamodule.setup()
    
    dataloader = datamodule.train_dataloader()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SCVIPerturbationModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False
    )
    
    # Diagnose
    results = diagnose_reconstruction_quality(model, dataloader, device=device, max_batches=args.max_batches)


if __name__ == '__main__':
    main()

