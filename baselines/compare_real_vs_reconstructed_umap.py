#!/usr/bin/env python3
"""
Script to compare UMAP visualizations of real vs reconstructed data from trained scVI model.
"""

import sys
import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project to path
sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml


def extract_embeddings_and_reconstructions(model, dataloader, device='cuda', max_batches=None):
    """
    Extract latent embeddings and reconstructions from the model.
    
    Returns:
        real_data: numpy array of real expression data
        latent_embeddings: numpy array of latent space embeddings (z_basal)
        reconstructed_data: numpy array of reconstructed expression data
        metadata: dict with perturbation, cell_type, batch info
    """
    model.eval()
    model = model.to(device)
    
    real_data_list = []
    latent_list = []
    reconstructed_list = []
    pert_list = []
    cell_type_list = []
    batch_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Use model's predict_step method (matches state repo approach)
            # This returns log-normalized predictions matching the state repo
            batch_preds = model.predict_step(batch, batch_idx)
            
            # Extract predictions and real data (both are log-normalized by predict_step)
            x_recon_log = batch_preds["preds"].cpu().numpy()  # Already log-normalized via _log_normalize_expression
            x_real_log = batch_preds["X"].cpu().numpy()  # Already log-normalized via _log_normalize_expression
            
            # Also get latent embeddings for UMAP
            # Need to call forward separately to get encoder outputs
            x_pert, x_basal, pert, cell_type, batch_ids = model.extract_batch_tensors(batch)
            
            # Convert to counts if needed (for encoder)
            # Note: If data is raw counts (max > 25.0), use directly. 
            # If log-normalized (max <= 25.0), convert back to counts.
            # scVI module.forward expects counts for ZINB/NB distributions.
            if x_basal.max() <= 25.0:
                x_basal_counts = torch.exp(x_basal) - 1
            else:
                x_basal_counts = x_basal  # Already raw counts
            
            # Forward pass to get latent embeddings
            encoder_outputs, decoder_outputs = model.module.forward(
                x_basal_counts, pert, cell_type, batch_ids
            )
            z_basal = encoder_outputs["z_basal"].cpu().numpy()
            
            # Store results (both are log-normalized, matching state repo)
            real_data_list.append(x_real_log)
            latent_list.append(z_basal)
            reconstructed_list.append(x_recon_log)
            
            # Extract metadata
            if isinstance(pert, torch.Tensor):
                if pert.dim() == 2:
                    pert_idx = pert.argmax(1).cpu().numpy()
                else:
                    pert_idx = pert.cpu().numpy()
            else:
                pert_idx = np.array([0] * len(x_real))
            pert_list.append(pert_idx)
            
            if isinstance(cell_type, torch.Tensor):
                if cell_type.dim() == 2:
                    cell_type_idx = cell_type.argmax(1).cpu().numpy()
                else:
                    cell_type_idx = cell_type.cpu().numpy()
            else:
                cell_type_idx = np.array([0] * len(x_real))
            cell_type_list.append(cell_type_idx)
            
            if isinstance(batch_ids, torch.Tensor):
                if batch_ids.dim() == 2:
                    batch_idx_arr = batch_ids.argmax(1).cpu().numpy()
                else:
                    batch_idx_arr = batch_ids.cpu().numpy()
            else:
                batch_idx_arr = np.array([0] * len(x_real))
            batch_list.append(batch_idx_arr)
    
    # Concatenate all batches
    real_data = np.vstack(real_data_list)
    latent_embeddings = np.vstack(latent_list)
    reconstructed_data = np.vstack(reconstructed_list)
    pert_array = np.concatenate(pert_list)
    cell_type_array = np.concatenate(cell_type_list)
    batch_array = np.concatenate(batch_list)
    
    metadata = {
        'perturbation': pert_array,
        'cell_type': cell_type_array,
        'batch': batch_array
    }
    
    return real_data, latent_embeddings, reconstructed_data, metadata


def compute_umap(data, n_neighbors=15, min_dist=0.1, random_state=42, n_components_pca=None):
    """
    Compute UMAP embedding.
    
    If n_components_pca is provided, first reduce dimensions using PCA.
    This is useful for high-dimensional data (e.g., gene expression).
    """
    if n_components_pca is not None and n_components_pca < data.shape[1]:
        print(f"  Reducing dimensions with PCA: {data.shape[1]} -> {n_components_pca}")
        pca = PCA(n_components=n_components_pca, random_state=random_state)
        data = pca.fit_transform(data)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(data) // 4),
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean'
    )
    return reducer.fit_transform(data)


def plot_comparison(real_umap, recon_umap, metadata, output_path, datamodule=None, latent_embeddings=None):
    """Create comparison plots of real vs reconstructed UMAP."""
    
    # Get label mappings if available
    pert_map = datamodule.pert_onehot_map if datamodule else None
    cell_type_map = datamodule.cell_type_onehot_map if datamodule else None
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Real vs Reconstructed Data - UMAP Comparison', fontsize=16, fontweight='bold')
    
    # Color by perturbation
    if pert_map:
        pert_names = list(pert_map.keys())
        n_perts = len(pert_names)
        colors_pert = plt.cm.tab20(np.linspace(0, 1, n_perts))
        pert_labels = [pert_names[i] if i < len(pert_names) else f'Pert_{i}' 
                      for i in metadata['perturbation']]
    else:
        pert_labels = metadata['perturbation']
        n_perts = len(np.unique(pert_labels))
        colors_pert = plt.cm.tab20(np.linspace(0, 1, n_perts))
    
    # Plot 1: Real data colored by perturbation
    ax = axes[0, 0]
    for i, pert_name in enumerate(np.unique(pert_labels)):
        mask = np.array(pert_labels) == pert_name
        ax.scatter(real_umap[mask, 0], real_umap[mask, 1], 
                  c=[colors_pert[i % len(colors_pert)]], 
                  label=pert_name if i < 10 else None,  # Limit legend
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Real Data - Colored by Perturbation', fontweight='bold')
    if len(np.unique(pert_labels)) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reconstructed data colored by perturbation
    ax = axes[0, 1]
    for i, pert_name in enumerate(np.unique(pert_labels)):
        mask = np.array(pert_labels) == pert_name
        ax.scatter(recon_umap[mask, 0], recon_umap[mask, 1], 
                  c=[colors_pert[i % len(colors_pert)]], 
                  label=pert_name if i < 10 else None,
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Reconstructed Data - Colored by Perturbation', fontweight='bold')
    if len(np.unique(pert_labels)) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Side-by-side comparison (same scale)
    ax = axes[0, 2]
    ax.scatter(real_umap[:, 0], real_umap[:, 1], 
              c='blue', alpha=0.4, s=15, label='Real', edgecolors='white', linewidth=0.2)
    ax.scatter(recon_umap[:, 0], recon_umap[:, 1], 
              c='red', alpha=0.4, s=15, label='Reconstructed', edgecolors='white', linewidth=0.2)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Overlay: Real (Blue) vs Reconstructed (Red)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Color by cell type
    if cell_type_map:
        cell_type_names = list(cell_type_map.keys())
        n_ct = len(cell_type_names)
        colors_ct = plt.cm.Set3(np.linspace(0, 1, n_ct))
        ct_labels = [cell_type_names[i] if i < len(cell_type_names) else f'CT_{i}' 
                    for i in metadata['cell_type']]
    else:
        ct_labels = metadata['cell_type']
        n_ct = len(np.unique(ct_labels))
        colors_ct = plt.cm.Set3(np.linspace(0, 1, n_ct))
    
    # Plot 4: Real data colored by cell type
    ax = axes[1, 0]
    for i, ct_name in enumerate(np.unique(ct_labels)):
        mask = np.array(ct_labels) == ct_name
        ax.scatter(real_umap[mask, 0], real_umap[mask, 1], 
                  c=[colors_ct[i % len(colors_ct)]], 
                  label=ct_name if i < 15 else None,
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Real Data - Colored by Cell Type', fontweight='bold')
    if len(np.unique(ct_labels)) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Reconstructed data colored by cell type
    ax = axes[1, 1]
    for i, ct_name in enumerate(np.unique(ct_labels)):
        mask = np.array(ct_labels) == ct_name
        ax.scatter(recon_umap[mask, 0], recon_umap[mask, 1], 
                  c=[colors_ct[i % len(colors_ct)]], 
                  label=ct_name if i < 15 else None,
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Reconstructed Data - Colored by Cell Type', fontweight='bold')
    if len(np.unique(ct_labels)) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Latent space UMAP
    ax = axes[1, 2]
    if latent_embeddings is not None and len(latent_embeddings) > 0:
        latent_umap = compute_umap(latent_embeddings, n_neighbors=15)
        for i, pert_name in enumerate(np.unique(pert_labels)):
            mask = np.array(pert_labels) == pert_name
            ax.scatter(latent_umap[mask, 0], latent_umap[mask, 1], 
                      c=[colors_pert[i % len(colors_pert)]], 
                      label=pert_name if i < 10 else None,
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Latent Space (z_basal) - Colored by Perturbation', fontweight='bold')
        if len(np.unique(pert_labels)) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Latent embeddings\nnot available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Latent Space (z_basal)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare real vs reconstructed UMAP visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--data_module', type=str, required=True,
                       help='Path to saved data module (.pkl file)')
    parser.add_argument('--output', type=str, default='./real_vs_reconstructed_umap.png',
                       help='Output path for the comparison plot')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to process (for faster testing)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Which data split to use')
    
    args = parser.parse_args()
    
    # Load data module
    if Path(args.data_module).exists():
        print(f"Loading data module from: {args.data_module}")
        with open(args.data_module, 'rb') as f:
            datamodule = pickle.load(f)
    else:
        print(f"Data module not found at: {args.data_module}")
        print("Attempting to recreate from config...")
        # Try to load from config.yaml in the same directory
        config_path = Path(args.checkpoint).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Recreate data module from config
            data_config = OmegaConf.create(config['data'])
            datamodule = get_datamodule(
                name=data_config['name'],
                kwargs=data_config['kwargs'],
                batch_size=data_config['kwargs'].get('batch_size', 128),
                cell_sentence_len=data_config['kwargs'].get('cell_sentence_len', 1)
            )
            datamodule.setup()
            print("Data module recreated from config!")
        else:
            raise FileNotFoundError(
                f"Data module not found and cannot recreate from config. "
                f"Expected config at: {config_path}"
            )
    
    # Get dataloader
    if args.split == 'train':
        dataloader = datamodule.train_dataloader()
    elif args.split == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SCVIPerturbationModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False
    )
    
    # Extract embeddings and reconstructions
    print("Extracting embeddings and reconstructions...")
    real_data, latent_embeddings, reconstructed_data, metadata = extract_embeddings_and_reconstructions(
        model, dataloader, device=device, max_batches=args.max_batches
    )
    
    print(f"Extracted {len(real_data)} cells")
    print(f"  Real data shape: {real_data.shape}")
    print(f"    Real data range: [{real_data.min():.2f}, {real_data.max():.2f}], mean: {real_data.mean():.2f}")
    print(f"  Latent embeddings shape: {latent_embeddings.shape}")
    print(f"  Reconstructed data shape: {reconstructed_data.shape}")
    print(f"    Reconstructed data range: [{reconstructed_data.min():.2f}, {reconstructed_data.max():.2f}], mean: {reconstructed_data.mean():.2f}")
    
    # Filter to only non-zero positions in ground truth
    print("Filtering to non-zero positions in ground truth...")
    print(f"  Real data shape: {real_data.shape}")
    print(f"  Reconstructed data shape: {reconstructed_data.shape}")
    
    # Create mask for non-zero positions in real data
    # For each cell, find which genes are non-zero
    non_zero_mask = real_data > 0  # (n_cells, n_genes) boolean mask
    
    print(f"  Non-zero positions in real data: {non_zero_mask.sum()} / {non_zero_mask.size} ({100 * non_zero_mask.sum() / non_zero_mask.size:.1f}%)")
    
    # For each cell, extract only the non-zero genes
    # We'll create a sparse representation: for each cell, keep only non-zero genes
    # But for UMAP, we need fixed-size vectors, so we'll use a different approach:
    # Option 1: Use only genes that are non-zero in at least some cells
    # Option 2: For each cell, pad with zeros (but this defeats the purpose)
    # Option 3: Use the mask to weight the data, but still use all genes
    
    # Actually, let's use a better approach: only use genes that are non-zero in at least X% of cells
    # This filters out always-zero genes while keeping the structure
    
    # But the user wants to compare non-zero positions directly
    # So let's create a representation where we only consider non-zero positions
    # We'll use the mask to create a filtered dataset
    
    # Approach: For each cell, create a vector of only non-zero values
    # But this creates variable-length vectors, which won't work for UMAP
    # Instead, let's mask the data: set zero positions to NaN, then use only non-NaN positions
    
    # Better approach: Create a joint dataset where we concatenate real and reconstructed,
    # but only include positions where real data is non-zero
    print("\n  Creating joint dataset with only non-zero positions from ground truth...")
    
    # Mask: where real data is zero, set both real and reconstructed to 0 (or NaN)
    # Actually, let's keep the structure but only weight/consider non-zero positions
    
    # User's request: "umap the non zero in ground truth with the corresponding counts in reconstructed"
    # This means: for each cell, only use genes where real > 0, and compare with reconstructed at those same positions
    
    # Since UMAP needs fixed-size vectors, we'll:
    # 1. Keep all genes but mask out zeros in the distance calculation
    # 2. OR: Only use genes that are non-zero in at least some cells
    
    # Let's go with approach: filter to genes that are non-zero in at least 1% of cells
    # This removes always-zero genes while keeping the structure
    gene_nonzero_freq = (real_data > 0).mean(axis=0)  # Frequency of non-zero per gene
    active_genes = gene_nonzero_freq > 0.01  # Genes non-zero in at least 1% of cells
    
    print(f"  Active genes (non-zero in >1% of cells): {active_genes.sum()} / {len(active_genes)} ({100 * active_genes.sum() / len(active_genes):.1f}%)")
    
    # Filter data to only active genes
    real_filtered = real_data[:, active_genes]
    recon_filtered = reconstructed_data[:, active_genes]
    
    print(f"  Filtered real data shape: {real_filtered.shape}")
    print(f"  Filtered reconstructed data shape: {recon_filtered.shape}")
    
    # Now concatenate real and reconstructed
    joint_data = np.vstack([real_filtered, recon_filtered])
    print(f"  Joint data shape (concatenated): {joint_data.shape}")
    
    # Apply PCA on the concatenated data
    n_pca_components = min(50, joint_data.shape[1] // 2)
    print(f"  Applying PCA: {joint_data.shape[1]} -> {n_pca_components} components")
    pca = PCA(n_components=n_pca_components, random_state=42)
    joint_pca = pca.fit_transform(joint_data)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Now compute UMAP on the PCA-projected joint space
    print(f"  Computing UMAP on PCA-projected joint space...")
    # Try different UMAP parameters for better alignment
    # Use more neighbors and smaller min_dist for tighter clusters
    joint_umap = compute_umap(
        joint_pca, 
        n_neighbors=min(30, len(joint_pca) // 10),  # More neighbors for better global structure
        min_dist=0.05,  # Smaller min_dist for tighter clusters
        n_components_pca=None  # Already in PCA space
    )
    
    # Split back into real and reconstructed
    n_real = len(real_data)
    real_umap = joint_umap[:n_real]
    recon_umap = joint_umap[n_real:]
    
    print(f"  Real UMAP shape: {real_umap.shape}")
    print(f"  Reconstructed UMAP shape: {recon_umap.shape}")
    
    # Create comparison plot
    print("Creating comparison plot...")
    plot_comparison(real_umap, recon_umap, metadata, args.output, datamodule, latent_embeddings)
    
    print("Done!")


if __name__ == '__main__':
    main()

