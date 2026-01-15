#!/usr/bin/env python3
"""
Analyze systematic biases in reconstruction to understand UMAP misalignment.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml


def analyze_bias(model, dataloader, device='cuda', max_batches=10):
    """Analyze systematic reconstruction biases."""
    model.eval()
    model = model.to(device)
    
    all_real = []
    all_recon = []
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_preds = model.predict_step(batch, batch_idx)
            x_real = batch_preds["X"].cpu().numpy()
            x_recon = batch_preds["preds"].cpu().numpy()
            
            # Compute per-cell errors
            errors = x_recon - x_real
            all_real.append(x_real)
            all_recon.append(x_recon)
            all_errors.append(errors)
    
    real = np.vstack(all_real)
    recon = np.vstack(all_recon)
    errors = np.vstack(all_errors)
    
    print("=" * 80)
    print("RECONSTRUCTION BIAS ANALYSIS")
    print("=" * 80)
    
    print(f"\n1. GLOBAL BIAS")
    print("-" * 80)
    print(f"Mean error (recon - real): {errors.mean():.4f}")
    print(f"Std error: {errors.std():.4f}")
    print(f"Mean absolute error: {np.abs(errors).mean():.4f}")
    
    # Check if reconstruction is systematically higher/lower
    positive_errors = (errors > 0).sum()
    negative_errors = (errors < 0).sum()
    print(f"Positive errors: {positive_errors} ({100*positive_errors/errors.size:.1f}%)")
    print(f"Negative errors: {negative_errors} ({100*negative_errors/errors.size:.1f}%)")
    
    print(f"\n2. PER-GENE BIAS")
    print("-" * 80)
    gene_mean_errors = errors.mean(axis=0)
    gene_std_errors = errors.std(axis=0)
    
    print(f"Genes with positive bias (recon > real): {(gene_mean_errors > 0.1).sum()}")
    print(f"Genes with negative bias (recon < real): {(gene_mean_errors < -0.1).sum()}")
    print(f"Genes with large bias (|mean error| > 0.5): {(np.abs(gene_mean_errors) > 0.5).sum()}")
    
    # Check if always-zero genes have systematic bias
    gene_sums = real.sum(axis=0)
    always_zero = gene_sums == 0
    if always_zero.sum() > 0:
        always_zero_errors = gene_mean_errors[always_zero]
        print(f"\nAlways-zero genes error analysis:")
        print(f"  Mean error: {always_zero_errors.mean():.4f}")
        print(f"  Max error: {always_zero_errors.max():.4f}")
        print(f"  Genes with error > 0.1: {(always_zero_errors > 0.1).sum()}")
    
    print(f"\n3. PER-CELL BIAS")
    print("-" * 80)
    cell_mean_errors = errors.mean(axis=1)
    cell_std_errors = errors.std(axis=1)
    
    print(f"Cells with positive bias: {(cell_mean_errors > 0.1).sum()}")
    print(f"Cells with negative bias: {(cell_mean_errors < -0.1).sum()}")
    print(f"Mean cell error: {cell_mean_errors.mean():.4f} ± {cell_mean_errors.std():.4f}")
    
    print(f"\n4. ERROR DISTRIBUTION IN PCA SPACE")
    print("-" * 80)
    # Project errors to PCA space
    joint_data = np.vstack([real, recon])
    pca = PCA(n_components=50, random_state=42)
    joint_pca = pca.fit_transform(joint_data)
    
    real_pca = joint_pca[:len(real)]
    recon_pca = joint_pca[len(real):]
    pca_errors = recon_pca - real_pca
    
    print(f"PCA space error magnitude: {np.linalg.norm(pca_errors, axis=1).mean():.4f}")
    print(f"PCA space error std: {np.linalg.norm(pca_errors, axis=1).std():.4f}")
    
    # Check which PCA components have largest errors
    pca_component_errors = np.abs(pca_errors).mean(axis=0)
    top_error_components = np.argsort(pca_component_errors)[-10:][::-1]
    print(f"Top 10 PCA components with largest errors: {top_error_components}")
    print(f"Their error magnitudes: {pca_component_errors[top_error_components]}")
    
    print(f"\n5. RECOMMENDATIONS")
    print("-" * 80)
    if errors.mean() > 0.1:
        print(f"⚠️  Systematic positive bias: reconstruction is systematically higher")
        print(f"   → Model may be predicting too many non-zeros")
    elif errors.mean() < -0.1:
        print(f"⚠️  Systematic negative bias: reconstruction is systematically lower")
        print(f"   → Model may be under-predicting expression")
    
    if always_zero.sum() > 0 and always_zero_errors.mean() > 0.1:
        print(f"⚠️  Always-zero genes have positive bias: {always_zero_errors.mean():.4f}")
        print(f"   → This is the main issue! Need stronger zero-gene penalty")
    
    if np.abs(pca_errors).mean() > 5.0:
        print(f"⚠️  Large errors in PCA space: {np.abs(pca_errors).mean():.4f}")
        print(f"   → Real and reconstructed are far apart in PCA space")
        print(f"   → This causes UMAP misalignment")
    
    print("\n" + "=" * 80)
    
    return {
        'real': real,
        'recon': recon,
        'errors': errors,
        'pca_errors': pca_errors
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze reconstruction bias')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_module', type=str, required=True)
    parser.add_argument('--max_batches', type=int, default=10)
    
    args = parser.parse_args()
    
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SCVIPerturbationModel.load_from_checkpoint(
        args.checkpoint, map_location=device, strict=False
    )
    
    analyze_bias(model, dataloader, device=device, max_batches=args.max_batches)


if __name__ == '__main__':
    main()

