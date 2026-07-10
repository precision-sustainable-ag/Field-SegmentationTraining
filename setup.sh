#!/usr/bin/env bash
# ==============================================================================
# setup.sh
#
# Bootstraps a blazingly fast, uv-managed Python environment for field segmentation.
# This replaces the legacy Conda environment.yaml workflow.
#
# Usage:
#   bash setup.sh                        # Uses defaults (Python 3.10, auto CUDA)
#   TORCH_CUDA=cu124 bash setup.sh       # Forces CUDA 12.4 PyTorch build
#   VENV_DIR=field_env bash setup.sh     # Uses a custom environment folder name
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Safety & Configuration
# ------------------------------------------------------------------------------
# Strict mode: The script will immediately crash if a command fails, an undefined 
# variable is used, or a pipeline fails. This prevents broken environments.
set -Eeuo pipefail

# Default environment variables. The ${VAR:-default} syntax allows you to 
# override these values in the terminal without editing this file.
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.field_segmentation}"
TORCH_CUDA="${TORCH_CUDA:-auto}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------

# Prints messages to the terminal with a standardized timestamp.
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Prints an error message and forcefully exits the script.
die() {
    echo "ERROR: $*" >&2
    exit 1
}

# This "trap" automatically catches any script failures and prints helpful 
# troubleshooting tips before exiting.
on_exit() {
    local exit_code=$?
    if [[ "$exit_code" -ne 0 ]]; then
        echo
        echo "=================================================="
        echo "Setup failed with exit code ${exit_code}."
        echo "Common fixes:"
        echo "  - Run this on a GPU node if nvidia-smi is unavailable."
        echo "  - Force a specific build with: TORCH_CUDA=cu124 bash setup.sh"
        echo "=================================================="
    fi
}
trap on_exit EXIT

# Checks if a necessary terminal command (like curl) is available.
require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# ------------------------------------------------------------------------------
# 3. Environment Setup Functions
# ------------------------------------------------------------------------------

# Checks for 'uv' (the ultra-fast Python package manager). If missing, it 
# downloads and installs it directly from the creators.
install_uv_if_missing() {
    if command -v uv >/dev/null 2>&1; then
        log "uv found: $(uv --version)"
        return
    fi

    log "uv not found. Installing uv into user space..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add standard uv install locations to the PATH so we can use it immediately.
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

    command -v uv >/dev/null 2>&1 || die "uv installation failed. Please install manually."
    log "uv successfully installed: $(uv --version)"
}

# Generates the list of required packages using flexible constraints.
# We do not pin exact versions here so 'uv' can mathematically resolve the 
# best, newest compatible combinations.
create_requirements_if_missing() {
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        log "Using existing ${REQUIREMENTS_FILE}"
        return
    fi

    log "Creating flexible ${REQUIREMENTS_FILE} for dependency resolution..."
    cat > "$REQUIREMENTS_FILE" <<'EOF'
# Core Data & Math
numpy
pandas

# Vision & Plotting
matplotlib
pillow
scikit-image
albumentations
opencv-python

# Utilities
tqdm
pyyaml
gputil

# Config & Tracking
hydra-core
omegaconf
wandb
fiftyone

# Segmentation & Metrics
pytorch-lightning
segmentation-models-pytorch
torchmetrics

# Optional / Future use
ultralytics
ollama
EOF
}

# ------------------------------------------------------------------------------
# 4. Hardware Detection
# ------------------------------------------------------------------------------

# Asks the system's Nvidia drivers what CUDA version it is running.
detect_cuda_version() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "none"
        return
    fi
    nvidia-smi \
        | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' \
        | head -n 1
}

# Translates the system CUDA version into the corresponding PyTorch wheel tag.
detect_torch_cuda() {
    local cuda_version
    cuda_version="$(detect_cuda_version)"

    case "$cuda_version" in
        12.6|12.7|12.8|12.9) echo "cu126" ;;
        12.4|12.5)           echo "cu124" ;;
        12.1|12.2|12.3)      echo "cu121" ;;
        none|"")             echo "cpu" ;;
        *)
            log "Unrecognized CUDA version '${cuda_version}'. Defaulting to cu126." >&2
            echo "cu126"
            ;;
    esac
}

# Maps the PyTorch wheel tag to the official Nvidia/PyTorch download URL.
torch_index_url_for() {
    local torch_cuda="$1"
    case "$torch_cuda" in
        cu126) echo "https://download.pytorch.org/whl/cu126" ;;
        cu124) echo "https://download.pytorch.org/whl/cu124" ;;
        cu121) echo "https://download.pytorch.org/whl/cu121" ;;
        cpu)   echo "https://download.pytorch.org/whl/cpu" ;;
        *)     die "Unsupported TORCH_CUDA='${torch_cuda}'. Use cu124, cu121, or cpu." ;;
    esac
}

# ------------------------------------------------------------------------------
# 5. Main Execution Engine
# ------------------------------------------------------------------------------

main() {
    log "Starting training environment setup..."

    # Ensure pre-requisites are met
    require_command curl
    install_uv_if_missing
    create_requirements_if_missing

    # Step A: Create and activate the isolated virtual environment
    log "Creating virtual environment: ${VENV_DIR}"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    
    log "Python environment active. Path: $(command -v python)"

    # Step B: Determine the correct hardware target for PyTorch
    if [[ "$TORCH_CUDA" == "auto" ]]; then
        DETECTED_CUDA_VERSION="$(detect_cuda_version)"
        TORCH_CUDA="$(detect_torch_cuda)"
        log "Auto-detected CUDA version: ${DETECTED_CUDA_VERSION}"
    fi
    TORCH_INDEX_URL="$(torch_index_url_for "$TORCH_CUDA")"
    log "Targeting PyTorch build: ${TORCH_CUDA} via ${TORCH_INDEX_URL}"

    # Step C: The Anchor (Install PyTorch FIRST)
    # We strictly pin PyTorch to v2.6.0 and install it from the custom Nvidia URL.
    # By anchoring this into the environment first, we prevent Ultralytics or 
    # PyTorch Lightning from accidentally clobbering it with CPU-only versions later.
    log "Anchoring PyTorch 2.6.0 to the environment..."
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio --index-url "$TORCH_INDEX_URL"

    # Step D: The Flexible Resolution (Install everything else)
    # uv reads the requirements.txt, sees PyTorch is already installed, accepts it 
    # as truth, and rapidly calculates the best versions for the remaining packages.
    log "Resolving and installing remaining dependencies..."
    uv pip install -r "$REQUIREMENTS_FILE"

    # Step E: Sanity Check
    # Run a quick, invisible Python script to ensure the GPU is actually accessible.
    log "Verifying installation..."
    python -c "
import torch
import ultralytics
import pytorch_lightning
import segmentation_models_pytorch
print(f'   -> PyTorch Version: {torch.__version__}')
print(f'   -> CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   -> GPU Count: {torch.cuda.device_count()}')
print('   -> Basic imports passed successfully!')
"

    # Step F: Version Control Lock
    # Snapshot the mathematically perfect environment uv just created so we can 
    # commit it to Git and guarantee reproducibility in the future.
    log "Writing exact dependency snapshot to uv.lock.txt..."
    uv pip freeze > uv.lock.txt

    # Success output
    cat <<EOF

==================================================
Setup successfully completed!
==================================================

To activate this environment, run:
  source ${VENV_DIR}/bin/activate

To commit this stable environment to version control:
  git add setup.sh requirements.txt uv.lock.txt
  git commit -m "chore: migrate to uv and lock environment dependencies"
EOF
}

# Execute the main function, passing along any terminal arguments
main "$@"