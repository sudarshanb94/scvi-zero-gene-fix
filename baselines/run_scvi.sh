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

# Install packages from requirements.txt, but handle the editable Git install separately
echo "Installing packages from requirements.txt..."
echo "Note: If installation fails, check the error messages above to see which package failed."

# Create a temporary requirements file without the editable Git line
grep -v "^-e git+" requirements.txt > /tmp/requirements_no_git.txt || cp requirements.txt /tmp/requirements_no_git.txt

# Install packages from the filtered requirements file
$PIP_BIN install -r /tmp/requirements_no_git.txt

# Install the cell-load package from Git separately (editable install)
echo "Installing cell-load from Git..."
$PIP_BIN install -e git+https://github.com/Arcinstitute/cell-load.git#egg=cell-load || \
$PIP_BIN install git+https://github.com/Arcinstitute/cell-load.git

# Clean up temp file
rm -f /tmp/requirements_no_git.txt

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
    # Verify installation
    if ! $PYTHON_BIN -c "import lightning" 2>/dev/null; then
        echo "Warning: pytorch-lightning installation may have failed, trying again..."
        $PIP_BIN install --force-reinstall pytorch-lightning==2.5.1.post0
    fi
fi

# Install wandb if missing
if ! $PYTHON_BIN -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    $PIP_BIN install wandb==0.20.1
fi

# Install transformers if missing
if ! $PYTHON_BIN -c "import transformers" 2>/dev/null; then
    echo "Installing transformers..."
    $PIP_BIN install transformers==4.52.4
fi

# Install torch-scatter if missing (needs torch to be installed first)
if ! $PYTHON_BIN -c "import torch_scatter" 2>/dev/null; then
    echo "Installing torch-scatter..."
    # First ensure torch is installed
    if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
        echo "torch is not installed, installing it first..."
        $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
    fi
    # Verify torch is actually importable
    if ! $PYTHON_BIN -c "import torch; print(f'torch version: {torch.__version__}')" 2>/dev/null; then
        echo "ERROR: torch is not importable even after installation"
        exit 1
    fi
    # Install torch-scatter with --no-build-isolation so it can see torch in the environment
    echo "Installing torch-scatter with --no-build-isolation (allows build to see torch)..."
    $PIP_BIN install torch-scatter==2.1.2 --no-build-isolation || {
        echo "ERROR: torch-scatter installation failed"
        echo "This may require manual installation or a different approach"
        exit 1
    }
fi

# Install torch-geometric if missing
if ! $PYTHON_BIN -c "import torch_geometric" 2>/dev/null; then
    echo "Installing torch-geometric..."
    # Ensure torch is installed first
    if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
        echo "torch is not installed, installing it first..."
        $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
    fi
    # Install torch-geometric (may need torch-scatter, which we just installed)
    $PIP_BIN install torch-geometric
fi

# Install dcor if missing
if ! $PYTHON_BIN -c "import dcor" 2>/dev/null; then
    echo "Installing dcor..."
    $PIP_BIN install dcor
fi

# Verify critical packages are installed one by one with better error messages
echo "Verifying critical packages..."
MISSING=()

if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
    MISSING+=("torch")
fi

if ! $PYTHON_BIN -c "import hydra" 2>/dev/null; then
    MISSING+=("hydra")
fi

if ! $PYTHON_BIN -c "import lightning" 2>/dev/null; then
    MISSING+=("lightning (pytorch-lightning)")
fi

if ! $PYTHON_BIN -c "import wandb" 2>/dev/null; then
    MISSING+=("wandb")
fi

if ! $PYTHON_BIN -c "import transformers" 2>/dev/null; then
    MISSING+=("transformers")
fi

if ! $PYTHON_BIN -c "import torch_scatter" 2>/dev/null; then
    MISSING+=("torch_scatter")
fi

if ! $PYTHON_BIN -c "import torch_geometric" 2>/dev/null; then
    MISSING+=("torch_geometric")
fi

if ! $PYTHON_BIN -c "import dcor" 2>/dev/null; then
    MISSING+=("dcor")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Error: Missing packages: ${MISSING[*]}"
    echo "Attempting to install missing packages..."
    for pkg in "${MISSING[@]}"; do
        case $pkg in
            "torch")
                echo "Installing torch..."
                $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
                ;;
            "hydra")
                echo "Installing hydra-core..."
                $PIP_BIN install hydra-core==1.3.2
                ;;
            "lightning (pytorch-lightning)")
                echo "Installing pytorch-lightning..."
                echo "Checking what's installed..."
                $PIP_BIN list | grep -i lightning || echo "No lightning packages found"
                echo "Uninstalling any existing lightning packages..."
                $PIP_BIN uninstall -y pytorch-lightning lightning 2>/dev/null || true
                echo "Installing pytorch-lightning==2.5.1.post0..."
                $PIP_BIN install pytorch-lightning==2.5.1.post0
                echo "Verifying installation..."
                $PIP_BIN show pytorch-lightning || echo "pytorch-lightning not found after install"
                echo "Testing import..."
                $PYTHON_BIN -c "import lightning; print('lightning imported successfully')" || \
                $PYTHON_BIN -c "import pytorch_lightning; print('pytorch_lightning imported successfully')" || \
                echo "Warning: Could not import lightning or pytorch_lightning"
                ;;
            "wandb")
                echo "Installing wandb..."
                $PIP_BIN install wandb==0.20.1
                ;;
            "transformers")
                echo "Installing transformers..."
                $PIP_BIN install transformers==4.52.4
                ;;
            "torch_scatter")
                echo "Installing torch-scatter..."
                # Ensure torch is installed first
                if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
                    echo "torch is not installed, installing it first..."
                    $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
                fi
                # Verify torch is actually importable
                if ! $PYTHON_BIN -c "import torch; print(f'torch version: {torch.__version__}')" 2>/dev/null; then
                    echo "ERROR: torch is not importable even after installation"
                    exit 1
                fi
                # Install torch-scatter with --no-build-isolation so it can see torch in the environment
                echo "Installing torch-scatter with --no-build-isolation (allows build to see torch)..."
                $PIP_BIN install torch-scatter==2.1.2 --no-build-isolation || {
                    echo "ERROR: torch-scatter installation failed"
                    echo "This may require manual installation or a different approach"
                    exit 1
                }
                ;;
            "torch_geometric")
                echo "Installing torch-geometric..."
                # Ensure torch is installed first
                if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
                    echo "torch is not installed, installing it first..."
                    $PIP_BIN install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || $PIP_BIN install torch==2.4.1
                fi
                # Install torch-geometric
                $PIP_BIN install torch-geometric
                ;;
            "dcor")
                echo "Installing dcor..."
                $PIP_BIN install dcor
                ;;
        esac
    done
    
    # Final verification - test each import individually to show which one fails
    echo "Testing imports individually..."
    $PYTHON_BIN -c "import torch" || { echo "ERROR: torch import failed"; exit 1; }
    $PYTHON_BIN -c "import hydra" || { echo "ERROR: hydra import failed"; exit 1; }
    
    # Test lightning import - try different variations
    if ! $PYTHON_BIN -c "import lightning" 2>/dev/null; then
        echo "ERROR: lightning import failed"
        echo "Trying to diagnose the issue..."
        echo "Installed packages with 'lightning' in name:"
        $PIP_BIN list | grep -i lightning || echo "No lightning packages found"
        echo "Trying alternative import:"
        $PYTHON_BIN -c "import pytorch_lightning; print('pytorch_lightning works')" || echo "pytorch_lightning also failed"
        echo "Reinstalling pytorch-lightning..."
        $PIP_BIN install --force-reinstall --no-cache-dir pytorch-lightning==2.5.1.post0
        $PYTHON_BIN -c "import lightning" || { 
            echo "ERROR: lightning still cannot be imported after reinstall"
            echo "Please check the pytorch-lightning installation manually"
            exit 1
        }
    fi
    
    $PYTHON_BIN -c "import wandb" || { echo "ERROR: wandb import failed"; exit 1; }
    $PYTHON_BIN -c "import transformers" || { echo "ERROR: transformers import failed"; exit 1; }
    $PYTHON_BIN -c "import torch_scatter" || { echo "ERROR: torch_scatter import failed"; exit 1; }
    $PYTHON_BIN -c "import torch_geometric" || { echo "ERROR: torch_geometric import failed"; exit 1; }
    $PYTHON_BIN -c "import dcor" || { echo "ERROR: dcor import failed"; exit 1; }
    echo "✓ All critical packages installed"
else
    echo "✓ All critical packages installed"
fi

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

