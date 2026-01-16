#!/bin/bash
# Script to run scVI with your data configuration

# Wandb authentication - set your API key here or export it before running
# Get your API key from: https://wandb.ai/authorize
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY='110e40307c967c072dbb2171ac8e7328924097d8'  # Replace with your actual API key
fi

cd baselines/

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH (it installs to $HOME/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"
    # Source the env file if it exists
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available after installation. Please check installation."
    exit 1
fi

# Create virtual environment with uv (installs Python if needed)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv --python 3.11 --seed
fi

# Activate the environment
source .venv/bin/activate

# Ensure we're using the venv's Python and pip
PYTHON_BIN=".venv/bin/python"
PIP_BIN=".venv/bin/pip"

# Verify Python is available
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python not found in venv at $PYTHON_BIN"
    exit 1
fi

# Install pip if it doesn't exist in the venv
if [ ! -f "$PIP_BIN" ]; then
    echo "Installing pip in venv..."
    $PYTHON_BIN -m ensurepip --upgrade || curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && $PYTHON_BIN get-pip.py && rm get-pip.py
fi

# Upgrade pip first
echo "Upgrading pip..."
$PIP_BIN install --upgrade pip

# Install all packages from requirements.txt using pip
# (uv pip doesn't support editable Git URLs, so we use regular pip)
echo "Installing packages from requirements.txt..."
echo "Note: If installation fails, check the error messages above to see which package failed."
$PIP_BIN install -r requirements.txt

# Install critical packages if they're missing (in case requirements.txt installation failed partially)
echo "Checking and installing critical packages..."

# Install torch if missing
if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
    echo "Installing torch..."
    $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
fi

# Install hydra-core if missing
if ! $PYTHON_BIN -c "import hydra" 2>/dev/null; then
    echo "Installing hydra-core..."
    $PIP_BIN install hydra-core==1.3.2
fi

# Install pytorch-lightning if missing
if ! $PYTHON_BIN -c "import lightning" 2>/dev/null; then
    echo "Installing pytorch-lightning..."
    $PIP_BIN install pytorch-lightning==2.5.1.post0
fi

# Install wandb if missing
if ! $PYTHON_BIN -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    $PIP_BIN install wandb==0.20.1
fi

# Verify critical packages are installed
echo "Verifying critical packages..."
$PYTHON_BIN -c "import torch; import hydra; import lightning; import wandb; print('✓ All critical packages installed')" || {
    echo "Error: Some critical packages are still missing. Please install them manually."
    exit 1
}

# Increase file descriptor limit for large datasets (978 files in train_hvg)
# Set to a high value to handle many open files during data loading
ulimit -n 262144 2>/dev/null || true

# Clear any stale CUDA contexts and reset CUDA state
$PYTHON_BIN -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('CUDA reset complete')" || echo "Warning: Could not clear CUDA cache (torch may not be installed or CUDA not available)"

# Set CUDA environment variables
# Don't restrict CUDA_VISIBLE_DEVICES - let PyTorch Lightning use all 8 GPUs


# Enable real-time wandb logging (removed offline mode)
# If you hit file handle issues, you can set: export WANDB_MODE=offline
# export WANDB_MODE=offline

# Run scVI training
$PYTHON_BIN -m state_sets_reproduce.train \
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

