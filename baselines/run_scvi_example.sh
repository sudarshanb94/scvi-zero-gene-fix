#!/bin/bash
# Example script to run scVI VAE training
# 
# Usage: 
#   bash run_scvi_example.sh <path_to_data_toml> <output_dir> <experiment_name>
#
# Example:
#   bash run_scvi_example.sh /path/to/your/data.toml ./outputs scvi_experiment_1

# Activate virtual environment
source .venv/bin/activate

# Get arguments or use defaults
DATA_TOML_PATH=${1:-"./data.toml"}
OUTPUT_DIR=${2:-"./outputs"}
EXPERIMENT_NAME=${3:-"scvi_run"}

# Set data parameters (adjust these based on your dataset)
BATCH_COL=${BATCH_COL:-"gem_group"}      # Column name for batch information
PERT_COL=${PERT_COL:-"gene"}              # Column name for perturbation information
CELL_TYPE_KEY=${CELL_TYPE_KEY:-"cell_line"}  # Column name for cell type
CONTROL_PERT=${CONTROL_PERT:-"non-targeting"}  # Control perturbation name

echo "Training scVI on dataset: $DATA_TOML_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"

# Run scVI training
python -m state_sets_reproduce.train \
    data.kwargs.toml_config_path=$DATA_TOML_PATH \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.num_workers=8 \
    data.kwargs.batch_col=${BATCH_COL} \
    data.kwargs.pert_col=${PERT_COL} \
    data.kwargs.cell_type_key=${CELL_TYPE_KEY} \
    data.kwargs.control_pert=${CONTROL_PERT} \
    training.max_steps=250000 \
    training.val_freq=5000 \
    training.test_freq=9000 \
    training.batch_size=128 \
    training.train_seed=42 \
    wandb=mohsen \
    wandb.tags="[scvi,${EXPERIMENT_NAME}]" \
    model=scvi \
    training=scvi \
    output_dir="${OUTPUT_DIR}" \
    name="${EXPERIMENT_NAME}" \
    use_wandb=false \
    overwrite=false

echo "Training completed! Checkpoints saved in: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/"

