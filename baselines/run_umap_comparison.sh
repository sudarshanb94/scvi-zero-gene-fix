#!/bin/bash
# Script to run UMAP comparison on trained scVI model
# 
# Usage:
#   bash run_umap_comparison.sh [checkpoint_path] [data_module_path] [output_path] [split]
#
# Example:
#   bash run_umap_comparison.sh ./outputs/scvi_experiment/checkpoints/final.ckpt ./outputs/scvi_experiment/data_module.pkl ./umap_comparison.png train

cd /work/baselines
source .venv/bin/activate

# Default values - use latest checkpoint
LATEST_CKPT=$(ls -t ./outputs/scvi_experiment/checkpoints/step=*.ckpt 2>/dev/null | head -1)
CHECKPOINT=${1:-${LATEST_CKPT:-"./outputs/scvi_experiment/checkpoints/final.ckpt"}}
DATA_MODULE=${2:-"./outputs/scvi_experiment/data_module.pkl"}
OUTPUT=${3:-"./real_vs_reconstructed_umap.png"}
SPLIT=${4:-"train"}

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints in ./outputs/scvi_experiment/checkpoints/:"
    ls -la ./outputs/scvi_experiment/checkpoints/ 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

if [ ! -f "$DATA_MODULE" ]; then
    echo "Error: Data module file not found: $DATA_MODULE"
    echo ""
    echo "Available data modules:"
    find ./outputs -name "data_module.pkl" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Running UMAP comparison..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Data module: $DATA_MODULE"
echo "  Output: $OUTPUT"
echo "  Split: $SPLIT"
echo ""

python compare_real_vs_reconstructed_umap.py \
    --checkpoint "$CHECKPOINT" \
    --data_module "$DATA_MODULE" \
    --output "$OUTPUT" \
    --split "$SPLIT"

echo ""
echo "Done! Check the output at: $OUTPUT"


