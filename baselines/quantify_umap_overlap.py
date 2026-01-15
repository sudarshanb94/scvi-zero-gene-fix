#!/usr/bin/env python3
"""
Quantify UMAP overlap between real and reconstructed data.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).parent))

from state_sets_reproduce.models import SCVIPerturbationModel
from cell_load.utils.modules import get_datamodule
from omegaconf import OmegaConf
import yaml
from compare_real_vs_reconstructed_umap import extract_embeddings_and_reconstructions, compute_umap
from sklearn.decomposition import PCA


def quantify_overlap(real_umap, recon_umap):
    """Quantify how well real and reconstructed UMAPs overlap."""
    
    print("=" * 80)
    print("UMAP OVERLAP QUANTIFICATION")
    print("=" * 80)
    
    # 1. Mean distance between corresponding points
    distances = np.linalg.norm(real_umap - recon_umap, axis=1)
    mean_distance = distances.mean()
    median_distance = np.median(distances)
    std_distance = distances.std()
    
    print(f"\n1. POINT-TO-POINT DISTANCES")
    print("-" * 80)
    print(f"Mean distance: {mean_distance:.4f}")
    print(f"Median distance: {median_distance:.4f}")
    print(f"Std distance: {std_distance:.4f}")
    print(f"Min distance: {distances.min():.4f}")
    print(f"Max distance: {distances.max():.4f}")
    
    # Percentiles
    p25 = np.percentile(distances, 25)
    p75 = np.percentile(distances, 75)
    print(f"25th percentile: {p25:.4f}")
    print(f"75th percentile: {p75:.4f}")
    
    # Cells with small distances (good overlap)
    threshold_good = mean_distance * 0.5  # Half of mean distance
    good_overlap = (distances < threshold_good).sum()
    print(f"\nCells with distance < {threshold_good:.4f} (good overlap): {good_overlap} / {len(distances)} ({100*good_overlap/len(distances):.1f}%)")
    
    # 2. Nearest neighbor analysis
    print(f"\n2. NEAREST NEIGHBOR ANALYSIS")
    print("-" * 80)
    # For each real point, find nearest reconstructed point
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(recon_umap)
    distances_to_nearest, indices = nbrs.kneighbors(real_umap)
    distances_to_nearest = distances_to_nearest.flatten()
    
    print(f"Mean distance to nearest recon point: {distances_to_nearest.mean():.4f}")
    print(f"Cells where nearest recon is the corresponding point: {(indices.flatten() == np.arange(len(real_umap))).sum()} / {len(real_umap)}")
    
    # 3. Centroid distance
    real_centroid = real_umap.mean(axis=0)
    recon_centroid = recon_umap.mean(axis=0)
    centroid_distance = np.linalg.norm(real_centroid - recon_centroid)
    
    print(f"\n3. CENTROID ANALYSIS")
    print("-" * 80)
    print(f"Real data centroid: {real_centroid}")
    print(f"Recon data centroid: {recon_centroid}")
    print(f"Centroid distance: {centroid_distance:.4f}")
    
    # 4. Variance comparison
    real_var = real_umap.var(axis=0).sum()
    recon_var = recon_umap.var(axis=0).sum()
    var_ratio = recon_var / real_var if real_var > 0 else 0
    
    print(f"\n4. VARIANCE ANALYSIS")
    print("-" * 80)
    print(f"Real data variance: {real_var:.4f}")
    print(f"Recon data variance: {recon_var:.4f}")
    print(f"Variance ratio (recon/real): {var_ratio:.4f}")
    
    # 5. Overlap score (inverse of mean distance, normalized)
    # Lower distance = better overlap
    umap_scale = np.std(real_umap)  # Scale of the UMAP embedding
    normalized_distance = mean_distance / umap_scale if umap_scale > 0 else mean_distance
    overlap_score = 1.0 / (1.0 + normalized_distance)  # Score between 0 and 1
    
    print(f"\n5. OVERLAP SCORE")
    print("-" * 80)
    print(f"UMAP scale (std): {umap_scale:.4f}")
    print(f"Normalized mean distance: {normalized_distance:.4f}")
    print(f"Overlap score (0-1, higher is better): {overlap_score:.4f}")
    
    if overlap_score > 0.7:
        print(f"  ✓ Excellent overlap!")
    elif overlap_score > 0.5:
        print(f"  ✓ Good overlap")
    elif overlap_score > 0.3:
        print(f"  ⚠️  Moderate overlap - room for improvement")
    else:
        print(f"  ⚠️  Poor overlap - significant misalignment")
    
    print("\n" + "=" * 80)
    
    return {
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'centroid_distance': centroid_distance,
        'overlap_score': overlap_score,
        'good_overlap_pct': 100 * good_overlap / len(distances)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quantify UMAP overlap')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_module', type=str, required=True)
    parser.add_argument('--max_batches', type=int, default=5)
    
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
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = SCVIPerturbationModel.load_from_checkpoint(
        args.checkpoint, map_location=device, strict=False
    )
    
    # Extract data
    real_data, latent_embeddings, reconstructed_data, metadata = \
        extract_embeddings_and_reconstructions(model, dataloader, device=device, max_batches=args.max_batches)
    
    # Filter to active genes and compute joint UMAP (same as comparison script)
    gene_nonzero_freq = (real_data > 0).mean(axis=0)
    active_genes = gene_nonzero_freq > 0.01
    real_filtered = real_data[:, active_genes]
    recon_filtered = reconstructed_data[:, active_genes]
    joint_data = np.vstack([real_filtered, recon_filtered])
    
    # Apply PCA
    pca = PCA(n_components=50, random_state=42)
    joint_pca = pca.fit_transform(joint_data)
    
    # Compute UMAP
    joint_umap = compute_umap(joint_pca, n_neighbors=15, n_components_pca=None)
    
    # Split
    n_real = len(real_data)
    real_umap = joint_umap[:n_real]
    recon_umap = joint_umap[n_real:]
    
    # Quantify
    results = quantify_overlap(real_umap, recon_umap)


if __name__ == '__main__':
    main()

