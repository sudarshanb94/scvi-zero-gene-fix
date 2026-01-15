#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
import pickle
import re
import gc
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import wandb
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

# Add baselines directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from scipy.sparse import csr_matrix
from scipy import stats
from tqdm import tqdm

from cell_load.mapping_strategies import (
    BatchMappingStrategy,
    RandomMappingStrategy,
)
from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule

torch.multiprocessing.set_sharing_strategy("file_system")


def remove_outliers(data, method='iqr', z_threshold=3.0, iqr_factor=1.5):
    """
    Remove outliers from data using various methods.
    
    Args:
        data: numpy array of shape (n_cells, n_genes)
        method: 'iqr' (Interquartile Range) or 'zscore'
        z_threshold: Z-score threshold for zscore method
        iqr_factor: IQR factor for IQR method
    
    Returns:
        mask: boolean array indicating which cells to keep
    """
    # Compute summary statistics per cell
    total_expr = data.sum(axis=1)  # Total expression per cell
    n_expressed = (data > 0).sum(axis=1)  # Number of expressed genes per cell
    
    if method == 'zscore':
        # Z-score based outlier detection
        z_scores_total = np.abs(stats.zscore(total_expr))
        z_scores_n_exp = np.abs(stats.zscore(n_expressed))
        mask = (z_scores_total < z_threshold) & (z_scores_n_exp < z_threshold)
    elif method == 'iqr':
        # IQR based outlier detection
        q1_total, q3_total = np.percentile(total_expr, [25, 75])
        iqr_total = q3_total - q1_total
        lower_total = q1_total - iqr_factor * iqr_total
        upper_total = q3_total + iqr_factor * iqr_total
        
        q1_n_exp, q3_n_exp = np.percentile(n_expressed, [25, 75])
        iqr_n_exp = q3_n_exp - q1_n_exp
        lower_n_exp = q1_n_exp - iqr_factor * iqr_n_exp
        upper_n_exp = q3_n_exp + iqr_factor * iqr_n_exp
        
        mask = ((total_expr >= lower_total) & (total_expr <= upper_total) &
                (n_expressed >= lower_n_exp) & (n_expressed <= upper_n_exp))
    else:
        mask = np.ones(len(data), dtype=bool)
    
    n_outliers = (~mask).sum()
    logger.info(f"  Outlier removal ({method}): removed {n_outliers} / {len(data)} cells ({100*n_outliers/len(data):.1f}%)")
    return mask


def create_umap_from_adata(adata_real, adata_pred, datamodule, output_path, use_count_space=False, remove_outliers_pred=True):
    """
    Create UMAP visualization comparing generated vs true cells from AnnData files.
    
    Args:
        remove_outliers_pred: If True, remove outliers from predicted data before UMAP
    """
    import matplotlib.pyplot as plt
    import umap
    from sklearn.decomposition import PCA
    
    logger.info("Creating UMAP from AnnData files...")
    
    # Extract expression data
    if hasattr(adata_real.X, 'toarray'):
        real_data = adata_real.X.toarray()
    else:
        real_data = adata_real.X
    
    if hasattr(adata_pred.X, 'toarray'):
        pred_data = adata_pred.X.toarray()
    else:
        pred_data = adata_pred.X
    
    # Convert to count space if requested
    if use_count_space:
        logger.info("  Converting to count space...")
        # If log-normalized, convert back to counts
        if real_data.max() <= 25.0:
            real_data = np.exp(real_data) - 1
        if pred_data.max() <= 25.0:
            pred_data = np.exp(pred_data) - 1
        logger.info(f"  Real data range: [{real_data.min():.2f}, {real_data.max():.2f}], mean: {real_data.mean():.2f}")
        logger.info(f"  Pred data range: [{pred_data.min():.2f}, {pred_data.max():.2f}], mean: {pred_data.mean():.2f}")
    else:
        logger.info("  Using log-normalized space")
        logger.info(f"  Real data range: [{real_data.min():.2f}, {real_data.max():.2f}], mean: {real_data.mean():.2f}")
        logger.info(f"  Pred data range: [{pred_data.min():.2f}, {pred_data.max():.2f}], mean: {pred_data.mean():.2f}")
    
    # Extract metadata before filtering (for real data)
    pert_names = adata_real.obs.get('pert_name', None)
    cell_types = adata_real.obs.get('celltype_name', None)
    
    # Extract metadata for predicted data before filtering
    pert_names_pred_full = adata_pred.obs.get('pert_name', None) if hasattr(adata_pred, 'obs') else None
    cell_types_pred_full = adata_pred.obs.get('celltype_name', None) if hasattr(adata_pred, 'obs') else None
    
    # Remove outliers from predicted data if requested
    if remove_outliers_pred:
        logger.info("  Removing outliers from generated cells...")
        outlier_mask = remove_outliers(pred_data, method='iqr', iqr_factor=1.5)
        pred_data = pred_data[outlier_mask]
        # Filter metadata for predicted data
        if pert_names_pred_full is not None:
            pert_names_pred = pert_names_pred_full[outlier_mask]
        else:
            pert_names_pred = None
        if cell_types_pred_full is not None:
            cell_types_pred = cell_types_pred_full[outlier_mask]
        else:
            cell_types_pred = None
        logger.info(f"  After outlier removal: {len(pred_data)} generated cells (removed {len(outlier_mask) - len(pred_data)})")
    else:
        pert_names_pred = pert_names_pred_full
        cell_types_pred = cell_types_pred_full
    
    # Use all genes (no filtering)
    logger.info(f"  Using all {real_data.shape[1]} genes for UMAP")
    
    # Concatenate real and predicted
    joint_data = np.vstack([real_data, pred_data])
    logger.info(f"  Joint data shape: {joint_data.shape}")
    
    # Apply PCA
    n_pca_components = min(50, joint_data.shape[1] // 2)
    logger.info(f"  Applying PCA: {joint_data.shape[1]} -> {n_pca_components} components")
    pca = PCA(n_components=n_pca_components, random_state=42)
    joint_pca = pca.fit_transform(joint_data)
    logger.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Compute UMAP
    logger.info("  Computing UMAP on PCA-projected joint space...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, len(joint_pca) // 10),
        min_dist=0.05,
        random_state=42,
        metric='euclidean'
    )
    joint_umap = reducer.fit_transform(joint_pca)
    
    # Split back
    n_real = len(real_data)
    real_umap = joint_umap[:n_real]
    pred_umap = joint_umap[n_real:]
    
    # Metadata already extracted above
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Generated vs True Cells - UMAP Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Overlay
    ax = axes[0, 0]
    ax.scatter(real_umap[:, 0], real_umap[:, 1], 
              c='blue', alpha=0.4, s=15, label='True', edgecolors='white', linewidth=0.2)
    ax.scatter(pred_umap[:, 0], pred_umap[:, 1], 
              c='red', alpha=0.4, s=15, label='Generated', edgecolors='white', linewidth=0.2)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Overlay: True (Blue) vs Generated (Red)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: True colored by perturbation
    ax = axes[0, 1]
    if pert_names is not None:
        unique_perts = np.unique(pert_names)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_perts)))
        for i, pert in enumerate(unique_perts[:20]):  # Limit to 20 for readability
            mask = pert_names == pert
            ax.scatter(real_umap[mask, 0], real_umap[mask, 1], 
                      c=[colors[i]], label=pert if i < 10 else None,
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    else:
        ax.scatter(real_umap[:, 0], real_umap[:, 1], c='blue', alpha=0.6, s=20)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('True Data - Colored by Perturbation', fontweight='bold')
    if pert_names is not None and len(unique_perts) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Generated colored by perturbation
    ax = axes[1, 0]
    pert_names_to_use = pert_names_pred if pert_names_pred is not None else pert_names
    if pert_names_to_use is not None and len(pert_names_to_use) == len(pred_umap):
        unique_perts_pred = np.unique(pert_names_to_use)
        colors_pred = plt.cm.tab20(np.linspace(0, 1, len(unique_perts_pred)))
        for i, pert in enumerate(unique_perts_pred[:20]):
            mask = pert_names_to_use == pert
            ax.scatter(pred_umap[mask, 0], pred_umap[mask, 1], 
                      c=[colors_pred[i]], label=pert if i < 10 else None,
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    else:
        ax.scatter(pred_umap[:, 0], pred_umap[:, 1], c='red', alpha=0.6, s=20)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Generated Data - Colored by Perturbation', fontweight='bold')
    if pert_names is not None and len(unique_perts) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Colored by cell type
    ax = axes[1, 1]
    cell_types_to_use = cell_types_pred if cell_types_pred is not None else cell_types
    if cell_types is not None:
        unique_cts = np.unique(cell_types)
        colors_ct = plt.cm.Set3(np.linspace(0, 1, len(unique_cts)))
        for i, ct in enumerate(unique_cts[:15]):
            mask = cell_types == ct
            ax.scatter(real_umap[mask, 0], real_umap[mask, 1], 
                      c=[colors_ct[i]], label=ct if i < 15 else None,
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        # Plot predicted with filtered cell types
        if cell_types_to_use is not None and len(cell_types_to_use) == len(pred_umap):
            for i, ct in enumerate(unique_cts[:15]):
                mask = cell_types_to_use == ct
                if mask.sum() > 0:
                    ax.scatter(pred_umap[mask, 0], pred_umap[mask, 1], 
                              c=[colors_ct[i]], alpha=0.6, s=20, edgecolors='white', linewidth=0.3, marker='x')
    else:
        ax.scatter(real_umap[:, 0], real_umap[:, 1], c='blue', alpha=0.6, s=20, label='True')
        ax.scatter(pred_umap[:, 0], pred_umap[:, 1], c='red', alpha=0.6, s=20, label='Generated', marker='x')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('True (dots) vs Generated (x) - Colored by Cell Type', fontweight='bold')
    if cell_types is not None and len(unique_cts) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved UMAP plot to {output_path}")
    plt.close()


def parse_args():
    """
    CLI for evaluation. The arguments mirror some of the old script_lightning/eval_lightning.py.
    """
    parser = argparse.ArgumentParser(
        description="Get predictions from a trained PerturbationModel."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to use for generation. Default is 'test'.",
    )
    parser.add_argument(
        "--create_umap",
        action="store_true",
        help="Create UMAP visualization comparing generated vs true cells",
    )
    parser.add_argument(
        "--use_count_space",
        action="store_true",
        help="Use raw count space instead of log-normalized for UMAP",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for faster generation/testing)",
    )

    return parser.parse_args()


def post_process_preds(batch_preds, model_class_name):
    if "cell_mask" in batch_preds:
        new_batch_preds = {}
        cell_mask = batch_preds["cell_mask"]  # (batch_size, )
        for k, v in batch_preds.items():
            if k == "cell_mask":
                continue
            elif isinstance(v, torch.Tensor):
                new_batch_preds[k] = v[cell_mask]
            elif isinstance(v, list) and v[0] is not None:
                new_batch_preds[k] = [v[i] for i in range(len(v)) if cell_mask[i]]
            else:
                new_batch_preds[k] = v
    else:
        new_batch_preds = batch_preds

    if "gene_mask" in batch_preds:
        gene_mask = (
            batch_preds["gene_mask"].detach().cpu().numpy()
        )  # (batch_size, num_genes)
        gene_mask = gene_mask[0, :]
        gene_mask = gene_mask.astype(bool)
    else:
        gene_mask = None

    return new_batch_preds, gene_mask


def load_config(cfg_path: str) -> dict:
    """
    Load config from the YAML file that was dumped during training.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)

    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith("step=") and "val_loss" not in f:
            # Extract the number between 'step=' and '.ckpt'
            match = re.search(r"step=(\d+)(?:-v\d+)?\.ckpt", f)
            if match:
                step_numbers.append(int(match.group(1)))

    if not step_numbers:
        raise ValueError("No checkpoint files found")

    # Get the maximum step number
    max_step = max(step_numbers)

    # Construct the checkpoint path
    checkpoint_path = os.path.join(directory, f"step={max_step}.ckpt")

    return checkpoint_path


def main():
    args = parse_args()

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])

    try:
        cell_sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
    except:
        cell_sentence_len = 1

    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # 3. Load the data module
    data_module = get_datamodule(
        name=cfg["data"]["name"],
        kwargs=cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=cell_sentence_len,
    )
    data_module.setup()

    # seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 4. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    if cfg["model"]["name"].lower() == "lowranklinear":
        if (
            cfg["model"]["kwargs"]["pert_emb"] == "identity"
        ):  # Use the identity matrix as the perturbation embeddings (one-hot encoding)
            cfg["model"]["kwargs"]["pert_emb_path"] = "identity"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "scgpt"
        ):  # scGPT: Genetic perturbation data
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = f"/large_storage/goodarzilab/userspace/mohsen/VCI-models/scGPT/scGPT_human/gene_embeddings.h5"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "tahoe_rdkit"
        ):  # Tahoe: Chemical perturbation data
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI/tahoe/tahoe_rdkit_embs.h5"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "gears_norman"
        ):  # Extract GEARS perturbation embeddings from the trained GEARS on Norman2019 dataset
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI-models/GEARS/gears_norman.h5"
        else:
            raise ValueError(
                f"Unknown perturbation embedding: {cfg['model']['kwargs']['pert_emb']}"
            )

        if (
            cfg["model"]["kwargs"]["gene_emb"] == "training_data"
        ):  # Use the training data as the gene embeddings
            # 1. Perform PCA on the training data
            raise NotImplementedError("PCA on training data is not implemented yet")
        elif (
            cfg["model"]["kwargs"]["gene_emb"] == "gears_norman"
        ):  # Extract GEARS gene embeddings from the trained GEARS on Norman2019 dataset
            cfg["model"]["kwargs"][
                "gene_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI-models/GEARS/gears_norman.h5"
        elif (
            cfg["model"]["kwargs"]["gene_emb"] == "scgpt"
        ):  # Extract scGPT's vocabulary embeddings
            cfg["model"]["kwargs"][
                "gene_emb_path"
            ] = f"/large_storage/goodarzilab/userspace/mohsen/VCI-models/scGPT/scGPT_human/gene_embeddings.h5"
        else:
            raise ValueError(f"Unknown gene embedding: {cfg['model']['gene_emb']}")

    # The model architecture is determined by the config
    model_class_name = cfg["model"]["name"]  # e.g. "EmbedSum" or "NeuralOT"
    model_kwargs = cfg["model"]["kwargs"]  # dictionary of hyperparams

    # Build the correct class
    if model_class_name.lower() == "cpa":
        from state_sets_reproduce.models.cpa import CPAPerturbationModel

        ModelClass = CPAPerturbationModel
    elif model_class_name.lower() in ["scgpt-genetic", "scgpt-chemical", "scgpt"]:
        from state_sets_reproduce.models.scgpt import scGPTForPerturbationModel

        ModelClass = scGPTForPerturbationModel
    elif model_class_name.lower() == "scvi":
        from state_sets_reproduce.models.scvi import SCVIPerturbationModel

        ModelClass = SCVIPerturbationModel
    elif model_class_name.lower() == "lowranklinear":
        from state_sets_reproduce.models.low_rank_linear import LowRankLinearModel

        ModelClass = LowRankLinearModel
    elif model_class_name.lower() == "gears":
        from state_sets_reproduce.models.gears import GEARSPerturbationModel

        ModelClass = GEARSPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()  # e.g. input_dim, output_dim, pert_dim
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        # "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        # other model_kwargs keys to pass along:
        **model_kwargs,
    }

    # load checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    if model_class_name.lower() in ["scgpt", "scgpt-genetic", "scgpt-chemical"]:
        model.to(torch.bfloat16)

    logger.info("Model loaded successfully.")

    baseline_models = [
        "scvi",
        "cpa",
        "lowranklinear",
        "scgpt-genetic",
        "scgpt-chemical",
        "scgpt",
    ]

    # 5. Run inference on test set
    if model_class_name.lower() in [
        "scvi",
        "cpa",
        "lowranklinear",
        "gears",
    ] or model_class_name.lower().startswith("scgpt"):
        if (
            "cell_sentence_len" in cfg["model"]["kwargs"]
            and cfg["model"]["kwargs"]["cell_sentence_len"] > 1
        ):
            data_module.cell_sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
        else:
            data_module.cell_sentence_len = 1
    else:
        data_module.cell_sentence_len = cfg["model"]["kwargs"][
            "transformer_backbone_kwargs"
        ]["n_positions"]

    # Get dataloader based on split
    if args.split == "train":
        dataloader = data_module.train_dataloader()
        logger.info(f"Using TRAIN dataloader for generation")
    elif args.split == "val":
        dataloader = data_module.val_dataloader()
        logger.info(f"Using VAL dataloader for generation")
    else:
        dataloader = data_module.test_dataloader()
        logger.info(f"Using TEST dataloader for generation")

    print(f"DEBUG: data_module.batch_size: {data_module.batch_size}")

    if dataloader is None:
        logger.warning(f"No {args.split} dataloader found. Exiting.")
        sys.exit(0)

    # num_cells = test_loader.batch_sampler.tot_num
    # output_dim = var_dims["output_dim"]
    # gene_dim = var_dims["gene_dim"]
    # hvg_dim = var_dims["hvg_dim"]

    logger.info(f"Generating predictions on {args.split} set using manual loop...")
    device = next(model.parameters()).device

    final_preds = []
    final_reals = []

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (
        data_module.embed_key is not None
        and cfg["data"]["kwargs"]["output_space"] == "all"
    )

    if store_raw_expression:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if cfg["data"]["kwargs"]["output_space"] == "gene":
            final_X_hvg = []
            final_gene_preds = []
        if cfg["data"]["kwargs"]["output_space"] == "all":
            final_X_hvg = []
            final_gene_preds = []
    else:
        # Otherwise, use lists for later concatenation.
        final_X_hvg = None
        final_gene_preds = None

    logger.info(f"Generating predictions on {args.split} set ...")

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_ctrl_cell_barcodes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Predicting", unit="batch")
        ):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                logger.info(f"Stopping after {args.max_batches} batches (as requested)")
                break
            
            # Move each tensor in the batch to the model's device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Get predictions
            with torch.autocast(device_type="cuda", enabled=True):
                if model_class_name.lower() in [
                    "scgpt",
                    "scgpt-genetic",
                    "scgpt-chemical",
                ]:
                    batch = {
                        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                if batch_preds["preds"] is None:
                    continue

                batch_preds, gene_mask = post_process_preds(
                    batch_preds, model_class_name
                )

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])

            # Handle ctrl_cell_barcode
            if batch_preds["ctrl_cell_barcode"] is not None and isinstance(
                batch_preds["ctrl_cell_barcode"], list
            ):
                all_ctrl_cell_barcodes.extend(batch_preds["ctrl_cell_barcode"])
            else:
                all_ctrl_cell_barcodes.append(batch_preds["ctrl_cell_barcode"])

            # Handle celltype_name
            if isinstance(batch_preds["cell_type"], list):
                all_celltypes.extend(batch_preds["cell_type"])
            elif isinstance(batch_preds["cell_type"], torch.Tensor):
                if batch_preds["cell_type"].ndim == 2:
                    all_celltypes.extend(
                        batch_preds["cell_type"].argmax(dim=1).cpu().numpy().tolist()
                    )  # backward compatibility
                else:
                    all_celltypes.extend(batch_preds["cell_type"].cpu().numpy())
            else:
                all_celltypes.append(batch_preds["cell_type"])

            # Handle gem_group
            if isinstance(batch_preds["batch_name"], list):
                all_gem_groups.extend(batch_preds["batch_name"])
            elif isinstance(batch_preds["batch_name"], torch.Tensor):
                if batch_preds["batch_name"].ndim == 2:
                    all_gem_groups.extend(
                        batch_preds["batch_name"].argmax(dim=1).cpu().numpy().tolist()
                    )  # backward compatibility
                else:
                    all_gem_groups.extend(batch_preds["batch_name"].cpu().numpy())
            else:
                all_gem_groups.append(batch_preds["batch_name"])

            batch_pred_np = batch_preds["preds"].float().cpu().numpy()
            batch_real_np = batch_preds["X"].float().cpu().numpy()
            batch_size = batch_pred_np.shape[0]

            # print(f"DEBUG: batch_pred_np.shape: {batch_pred_np.shape}, batch_real_np.shape: {batch_real_np.shape}")

            final_preds.append(batch_pred_np)
            final_reals.append(batch_real_np)

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["X_hvg"].cpu().numpy()
                final_X_hvg.append(batch_real_gene_np)

            # Handle decoded gene predictions if available
            if final_gene_preds is not None:
                batch_gene_pred_np = batch_preds["gene_preds"].cpu().numpy()
                final_gene_preds.append(batch_gene_pred_np)

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs
    obs_data = {
        "pert_name": all_pert_names,
        "celltype_name": all_celltypes,
        "gem_group": all_gem_groups,
    }

    if len(all_ctrl_cell_barcodes) > 0:
        obs_data["ctrl_cell_barcode"] = all_ctrl_cell_barcodes

    obs = pd.DataFrame(obs_data)

    final_preds = np.concatenate(final_preds, axis=0)
    final_reals = np.concatenate(final_reals, axis=0)

    print(
        f"DEBUG: obs.shape: {obs.shape}, final_preds.shape: {final_preds.shape}, final_reals.shape: {final_reals.shape}"
    )

    # Create adata for predictions
    adata_pred = anndata.AnnData(X=final_preds, obs=obs)
    # Create adata for real
    adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Create adata for real data in gene space (if available)
    adata_real_gene = None
    if (
        final_X_hvg and len(final_X_hvg) > 0
    ):  # either this is available, or we are already working in gene space
        final_X_hvg = np.concatenate(final_X_hvg, axis=0)
        if "int_counts" in data_module.__dict__ and data_module.int_counts:
            final_X_hvg = np.log1p(final_X_hvg)
        adata_real_gene = anndata.AnnData(X=final_X_hvg, obs=obs)

    # Create adata for gene predictions (if available)
    adata_pred_gene = None
    if final_gene_preds and len(final_gene_preds) > 0:
        final_gene_preds = np.concatenate(final_gene_preds, axis=0)
        if "int_counts" in data_module.__dict__ and data_module.int_counts:
            final_gene_preds = np.log1p(final_gene_preds)
        adata_pred_gene = anndata.AnnData(X=final_gene_preds, obs=obs)

    # save out adata_real to the output directory (with split suffix)
    adata_real_out = os.path.join(args.output_dir, f"adata_real_{args.split}.h5ad")
    adata_real.write_h5ad(adata_real_out)
    logger.info(f"Saved adata_real to {adata_real_out}")

    adata_pred_out = os.path.join(args.output_dir, f"adata_pred_{args.split}.h5ad")
    adata_pred.write_h5ad(adata_pred_out)
    logger.info(f"Saved adata_pred to {adata_pred_out}")

    if adata_real_gene is not None:
        adata_real_gene_out = os.path.join(args.output_dir, f"adata_real_gene_{args.split}.h5ad")
        adata_real_gene.write_h5ad(adata_real_gene_out)
        logger.info(f"Saved adata_real_gene to {adata_real_gene_out}")

    if adata_pred_gene is not None:
        adata_pred_gene_out = os.path.join(args.output_dir, f"adata_pred_gene_{args.split}.h5ad")
        adata_pred_gene.write_h5ad(adata_pred_gene_out)
        logger.info(f"Saved adata_pred_gene to {adata_pred_gene_out}")

    # Create UMAP visualization if requested
    if args.create_umap:
        logger.info("Creating UMAP visualization...")
        # First UMAP without outlier removal
        create_umap_from_adata(
            adata_real, 
            adata_pred, 
            data_module,
            output_path=os.path.join(args.output_dir, f"umap_generated_{args.split}.png"),
            use_count_space=args.use_count_space,
            remove_outliers_pred=False
        )
        # Second UMAP with outlier removal
        logger.info("Creating UMAP visualization (with outlier removal)...")
        create_umap_from_adata(
            adata_real, 
            adata_pred, 
            data_module,
            output_path=os.path.join(args.output_dir, f"umap_generated_{args.split}_no_outliers.png"),
            use_count_space=args.use_count_space,
            remove_outliers_pred=True
        )


if __name__ == "__main__":
    main()
