# Summary of Changes for scVI Training

## 1. Zero-Gene Penalty Fix (Similar to GEARS Issue)

**File**: `state_sets_reproduce/models/scvi/_module.py` (line 314)

**Changes**:
- Increased penalty weight from `1.0` → `10.0` (10x stronger)
- Changed from L1 to L2 penalty for smoother gradients
- Added max penalty term to prevent outliers
- Minimum weight of 0.5 to ensure penalty is always applied

**Code**:
```python
zero_penalty_weight = max(0.5, 10.0 * (always_zero_mask.sum().float() / x_pert.shape[1]))
recon_loss = recon_loss + zero_penalty_weight * (zero_penalty_l2 + 0.1 * zero_penalty_max)
```

**Purpose**: Constrain always-zero genes to be predicted as zero, preventing pollution of reconstruction space.

## 2. Batch Conditioning: Donor ID

**File**: `run_scvi.sh` (line 14)

**Change**: 
- `batch_col=sample_name` → `batch_col=donor_id`

**Purpose**: Condition on donor-specific batch effects to account for donor variation.

## 3. Cell Type: Timepoint

**File**: `run_scvi.sh` (line 16)

**Change**: 
- `cell_type_key=suspension_enriched_cell_types` → `cell_type_key=experimental_perturbation_time_point`

**Purpose**: Treat timepoint as cell type since different timepoints represent varying levels of treatment exposure.

## 4. Reconstruction Extraction Fix

**File**: `compare_real_vs_reconstructed_umap.py`

**Changes**:
- Use `model.predict_step()` (matches state repo approach)
- Use `_log_normalize_expression()` for consistent normalization
- Filter to active genes (non-zero in >1% of cells) before UMAP
- Compute UMAP on joint space with PCA preprocessing

**Purpose**: Ensure real and reconstructed data are in the same space for fair comparison.

## 5. New Experiment Name

**File**: `run_scvi.sh` (line 25)

**Change**: 
- `name="scvi_experiment_fixed"` → `name="scvi_experiment_donor_timepoint"`

**Purpose**: Start fresh experiment with all the new changes.

## Expected Improvements

1. **Better zero-gene handling**: Fewer spurious non-zero predictions
2. **Better batch correction**: Donor-specific effects accounted for
3. **Better timepoint modeling**: Timepoint treated as distinct cell type
4. **Better UMAP overlap**: Improved reconstruction quality should lead to better alignment

## To Run

```bash
bash run_scvi.sh
```

This will train a new model with all the improvements.

