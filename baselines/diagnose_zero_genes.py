#!/usr/bin/env python3
"""
Diagnostic script to check if scVI is properly handling always-zero genes.
Compares to the GEARS masking issue.
"""

import sys
import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml


def analyze_zero_genes(model, dataloader, device='cuda', max_batches=10):
    """Analyze how the model handles always-zero genes."""
    model.eval()
    model = model.to(device)
    
    all_real = []
    all_recon = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Use model's forward method (matches state repo) to get raw predictions
            # Note: Data is raw counts, so max() will be > 25.0, and we use directly
            x_pert, x_basal, pert, cell_type, batch_ids = model.extract_batch_tensors(batch)
            
            # Convert to counts if needed (for encoder)
            # If data is raw counts (max > 25.0), use directly.
            # If log-normalized (max <= 25.0), convert back to counts.
            # scVI module.forward expects counts for ZINB/NB distributions.
            if x_basal.max() <= 25.0:
                x_basal_counts = torch.exp(x_basal) - 1
            else:
                x_basal_counts = x_basal  # Already raw counts
            
            # Forward pass to get raw reconstruction (in count space)
            encoder_outputs, decoder_outputs = model.module.forward(
                x_basal_counts, pert, cell_type, batch_ids
            )
            
            # Get reconstruction in count space (before normalization)
            if model.recon_loss == "gauss":
                x_recon = decoder_outputs["px"].loc.cpu().numpy()
            else:
                x_recon = decoder_outputs["px"].mu.cpu().numpy()
            
            # Get real data in count space
            # If data is raw counts (max > 25.0), use directly.
            # If log-normalized (max <= 25.0), convert back to counts.
            if x_pert.max() <= 25.0:
                x_real = (torch.exp(x_pert) - 1).cpu().numpy()
            else:
                x_real = x_pert.cpu().numpy()  # Already raw counts
            
            all_real.append(x_real)
            all_recon.append(x_recon)
    
    # Concatenate
    real_data = np.vstack(all_real)  # (n_cells, n_genes)
    recon_data = np.vstack(all_recon)
    
    # Find always-zero genes (genes that are zero in ALL cells)
    gene_sums = real_data.sum(axis=0)  # Sum across cells for each gene
    always_zero_genes = gene_sums == 0
    
    # Find sometimes-zero genes (zero in some cells but not all)
    sometimes_zero_genes = (real_data == 0).any(axis=0) & ~always_zero_genes
    
    # Find never-zero genes
    never_zero_genes = (real_data > 0).all(axis=0)
    
    print(f"Total genes: {len(gene_sums)}")
    print(f"Always-zero genes: {always_zero_genes.sum()} ({100*always_zero_genes.sum()/len(gene_sums):.1f}%)")
    print(f"Sometimes-zero genes: {sometimes_zero_genes.sum()} ({100*sometimes_zero_genes.sum()/len(gene_sums):.1f}%)")
    print(f"Never-zero genes: {never_zero_genes.sum()} ({100*never_zero_genes.sum()/len(gene_sums):.1f}%)")
    
    # Check predictions for always-zero genes
    if always_zero_genes.sum() > 0:
        always_zero_recon = recon_data[:, always_zero_genes]
        always_zero_real = real_data[:, always_zero_genes]
        
        print(f"\n=== Always-Zero Genes Analysis ===")
        print(f"Real data (should all be 0):")
        print(f"  Min: {always_zero_real.min():.6f}, Max: {always_zero_real.max():.6f}")
        print(f"  Mean: {always_zero_real.mean():.6f}")
        print(f"  Non-zero count: {(always_zero_real > 0).sum()}")
        
        print(f"\nReconstructed data:")
        print(f"  Min: {always_zero_recon.min():.6f}, Max: {always_zero_recon.max():.6f}")
        print(f"  Mean: {always_zero_recon.mean():.6f}")
        print(f"  Non-zero count: {(always_zero_recon > 1e-6).sum()} (threshold: 1e-6)")
        print(f"  Non-zero percentage: {100*(always_zero_recon > 1e-6).sum() / always_zero_recon.size:.2f}%")
        
        # Check per-gene statistics
        gene_means = always_zero_recon.mean(axis=0)
        gene_maxs = always_zero_recon.max(axis=0)
        
        print(f"\nPer-gene statistics (always-zero genes):")
        print(f"  Mean prediction per gene: min={gene_means.min():.6f}, max={gene_means.max():.6f}, mean={gene_means.mean():.6f}")
        print(f"  Max prediction per gene: min={gene_maxs.min():.6f}, max={gene_maxs.max():.6f}, mean={gene_maxs.mean():.6f}")
        print(f"  Genes with mean > 0.01: {(gene_means > 0.01).sum()}")
        print(f"  Genes with max > 0.1: {(gene_maxs > 0.1).sum()}")
        
        # This is the potential issue: if always-zero genes have high predictions
        problematic_genes = gene_means > 0.01
        if problematic_genes.sum() > 0:
            print(f"\n⚠️  WARNING: {problematic_genes.sum()} always-zero genes have mean prediction > 0.01")
            print(f"   This suggests the model isn't properly constrained for these genes!")
            print(f"   Similar to the GEARS masking issue.")
        else:
            print(f"\n✓ Always-zero genes are properly constrained (mean predictions < 0.01)")
    
    # Check sometimes-zero genes
    if sometimes_zero_genes.sum() > 0:
        sometimes_zero_recon = recon_data[:, sometimes_zero_genes]
        sometimes_zero_real = real_data[:, sometimes_zero_genes]
        
        # For cells where real is zero, check if recon is also small
        zero_mask = sometimes_zero_real == 0
        recon_at_zeros = sometimes_zero_recon[zero_mask]
        
        print(f"\n=== Sometimes-Zero Genes Analysis ===")
        print(f"Cells where real=0: {zero_mask.sum()}")
        print(f"Reconstruction at zero positions:")
        print(f"  Min: {recon_at_zeros.min():.6f}, Max: {recon_at_zeros.max():.6f}")
        print(f"  Mean: {recon_at_zeros.mean():.6f}")
        print(f"  Non-zero count: {(recon_at_zeros > 1e-6).sum()}")
    
    # Check never-zero genes
    if never_zero_genes.sum() > 0:
        never_zero_recon = recon_data[:, never_zero_genes]
        never_zero_real = real_data[:, never_zero_genes]
        
        print(f"\n=== Never-Zero Genes Analysis ===")
        print(f"Real data range: [{never_zero_real.min():.2f}, {never_zero_real.max():.2f}]")
        print(f"Reconstructed data range: [{never_zero_recon.min():.2f}, {never_zero_recon.max():.2f}]")
        mse = ((never_zero_recon - never_zero_real) ** 2).mean()
        print(f"MSE: {mse:.6f}")
    
    return {
        'always_zero_genes': always_zero_genes,
        'sometimes_zero_genes': sometimes_zero_genes,
        'never_zero_genes': never_zero_genes,
        'real_data': real_data,
        'recon_data': recon_data
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose zero gene handling in scVI')
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
    
    # Analyze
    results = analyze_zero_genes(model, dataloader, device=device, max_batches=args.max_batches)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("If always-zero genes have high predictions (>0.01), this indicates")
    print("a similar issue to GEARS - the model isn't properly constrained")
    print("to output zeros for genes that should always be zero.")
    print("\nThis could explain poor UMAP overlap if the reconstruction")
    print("space is polluted with spurious non-zero predictions.")


if __name__ == '__main__':
    main()

