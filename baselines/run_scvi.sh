#!/bin/bash
# Script to run scVI with your data configuration

# Wandb authentication - set your API key here or export it before running
# Get your API key from: https://wandb.ai/authorize
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY='110e40307c967c072dbb2171ac8e7328924097d8'  # Replace with your actual API key
fi

cd /work/baselines
source .venv/bin/activate

# Increase file descriptor limit for large datasets (978 files in train_hvg)
# Set to a high value to handle many open files during data loading
ulimit -n 262144 2>/dev/null || true

# Clear any stale CUDA contexts and reset CUDA state
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('CUDA reset complete')" 2>/dev/null || true

# Set CUDA environment variables
# Don't restrict CUDA_VISIBLE_DEVICES - let PyTorch Lightning use all 8 GPUs


# Enable real-time wandb logging (removed offline mode)
# If you hit file handle issues, you can set: export WANDB_MODE=offline
# export WANDB_MODE=offline

# Run scVI training
python -m state_sets_reproduce.train \
    data.kwargs.toml_config_path=/work/baselines/my_data_config.toml \
    data.kwargs.embed_key=null \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.num_workers=8 \
    data.kwargs.batch_col=donor_id \
    data.kwargs.pert_col=guide_target_gene_symbol \
    data.kwargs.cell_type_key=experimental_perturbation_time_point \
    data.kwargs.control_pert=NTC \
    training.max_steps=3906200 \
    training.n_epochs_kl_warmup=200 \
    +training.devices=8 \
    +training.strategy=ddp_spawn \
    training.val_freq=97656 \
    training.test_freq=97656 \
    training.batch_size=128 \
    model=scvi \
    training=scvi \
    output_dir='/mnt/czi-sci-ai/project-scg-llm-data-2/experiments/' \
    name='marson_scvi_vae' \
    use_wandb=true \
    wandb.entity=sud \
    overwrite=false

