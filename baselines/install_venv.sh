#!/bin/bash
# Install virtual environment from requirements.txt
# This script sets up a venv and installs all dependencies

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

# Local venv path in current directory
VENV_PATH=".venv"

echo "Setting up virtual environment at: $VENV_PATH"

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment with uv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    uv venv --python 3.12 --seed "$VENV_PATH"  # --seed includes pip (use 3.12 to match local venv)
else
    echo "Virtual environment already exists at $VENV_PATH"
    echo "Skipping creation. To recreate, delete the directory first."
fi

# Activate the environment
source "$VENV_PATH/bin/activate"

# Set pip cache directory to a writable location to avoid permission warnings
export PIP_CACHE_DIR="$(dirname "$0")/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# Set Python path
PYTHON_BIN="$VENV_PATH/bin/python"

# Use python -m pip instead of bin/pip (more reliable with uv venv)
PIP_CMD="$PYTHON_BIN -m pip"

# Ensure pip is available
if ! $PYTHON_BIN -m pip --version &> /dev/null; then
    echo "Installing pip in venv..."
    $PYTHON_BIN -m ensurepip --upgrade || {
        echo "ensurepip failed, trying alternative method..."
        curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        $PYTHON_BIN /tmp/get-pip.py
        rm -f /tmp/get-pip.py
    }
fi

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."

# Handle torch-scatter and flash-attn separately (needs torch installed first)
# Also exclude cell-load from requirements.txt since we install it from local directory
# Create a temporary requirements file without torch-scatter, flash-attn, and cell-load
TEMP_REQ="/tmp/requirements_no_special_$$.txt"
grep -v "^-e git+" requirements.txt | \
    grep -v "^torch-scatter" | \
    grep -v "^flash-attn" | \
    grep -v "cell-load" > "$TEMP_REQ" || cp requirements.txt "$TEMP_REQ"

# Install packages from the filtered requirements file (this will install torch)
echo "Installing standard packages (including torch)..."
$PIP_CMD install -r "$TEMP_REQ"

# Install torch-scatter separately after torch is installed
# Use --no-build-isolation so it can see torch in the environment
echo "Installing torch-scatter (requires torch to be installed first)..."
if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
    echo "ERROR: torch is not installed. Cannot install torch-scatter."
    exit 1
fi
$PIP_CMD install torch-scatter==2.1.2 --no-build-isolation

# Install cell-load from local directory (editable install)
# The local cell-load directory should have the var/gene_ids fallback patch already applied
CELL_LOAD_DIR="$(dirname "$0")/cell-load"
if [ -d "$CELL_LOAD_DIR" ]; then
    echo "Installing cell-load from local directory: $CELL_LOAD_DIR"
    $PIP_CMD install -e "$CELL_LOAD_DIR"
else
    echo "ERROR: cell-load directory not found at $CELL_LOAD_DIR"
    echo "Please ensure the cell-load directory exists in the baselines directory."
    echo "It should be cloned from: https://github.com/Arcinstitute/cell-load.git"
    exit 1
fi

# Clean up temp file
rm -f "$TEMP_REQ"

# Verify critical packages are installed
echo ""
echo "Verifying critical packages..."
MISSING=0

check_package() {
    if $PYTHON_BIN -c "import $1" 2>/dev/null; then
        echo "✓ $1"
        return 0
    else
        echo "✗ $1 - MISSING"
        return 1
    fi
}

check_package torch || MISSING=1
check_package hydra || MISSING=1
check_package lightning || MISSING=1
check_package wandb || MISSING=1
check_package transformers || MISSING=1
check_package torch_scatter || MISSING=1
check_package torch_geometric || MISSING=1
check_package dcor || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Warning: Some critical packages are missing!"
    echo "Attempting to install missing packages..."
    
    # Install missing packages individually
    ! $PYTHON_BIN -c "import torch" 2>/dev/null && $PIP_CMD install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || true
    ! $PYTHON_BIN -c "import hydra" 2>/dev/null && $PIP_CMD install hydra-core==1.3.2 || true
    ! $PYTHON_BIN -c "import lightning" 2>/dev/null && $PIP_CMD install pytorch-lightning==2.5.1.post0 || true
    ! $PYTHON_BIN -c "import wandb" 2>/dev/null && $PIP_CMD install wandb==0.20.1 || true
    ! $PYTHON_BIN -c "import transformers" 2>/dev/null && $PIP_CMD install transformers==4.52.4 || true
    
    # torch-scatter needs special handling - install without version suffix
    if ! $PYTHON_BIN -c "import torch_scatter" 2>/dev/null; then
        echo "Installing torch-scatter (requires torch to be installed first)..."
        # First ensure torch is installed
        if ! $PYTHON_BIN -c "import torch" 2>/dev/null; then
            $PIP_CMD install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || \
            $PIP_CMD install torch==2.4.1
        fi
        # Install torch-scatter without the build suffix
        $PIP_CMD install torch-scatter==2.1.2 --no-build-isolation || true
    fi
    
    ! $PYTHON_BIN -c "import torch_geometric" 2>/dev/null && $PIP_CMD install torch-geometric || true
    ! $PYTHON_BIN -c "import dcor" 2>/dev/null && $PIP_CMD install dcor || true
    
    echo ""
    echo "Re-verifying packages..."
    check_package torch || echo "ERROR: torch still missing"
    check_package hydra || echo "ERROR: hydra still missing"
    check_package lightning || echo "ERROR: lightning still missing"
    check_package wandb || echo "ERROR: wandb still missing"
    check_package transformers || echo "ERROR: transformers still missing"
    check_package torch_scatter || echo "ERROR: torch_scatter still missing"
    check_package torch_geometric || echo "ERROR: torch_geometric still missing"
    check_package dcor || echo "ERROR: dcor still missing"
else
    echo ""
    echo "✓ All critical packages are installed!"
fi

echo ""
echo "Installation complete!"
echo "Virtual environment is ready at: $VENV_PATH"
echo ""
echo "To use this venv, activate it with:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Or update your run script to use:"
echo "  source $VENV_PATH/bin/activate"
