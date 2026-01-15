#!/bin/bash
# Script to run scVI with your data configuration

cd /work/baselines
source .venv/bin/activate

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
    training.max_steps=250000 \
    training.val_freq=5000 \
    training.test_freq=9000 \
    training.batch_size=128 \
    model=scvi \
    training=scvi \
    output_dir="./outputs" \
    name="scvi_experiment_zi_temperature_zero_loss_v3" \
    use_wandb=false \
    overwrite=false

