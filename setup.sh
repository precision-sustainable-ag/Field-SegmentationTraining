#!/usr/bin/env bash
# ==============================================================================
# setup.sh
#
# Creates a uv-managed Python environment with host-aware storage behavior.
#
# SCINet hosts:
#   - CERES
#   - ATLAS
#
# On SCINet, large files, caches, Python installations, and environments are
# redirected to:
#
#   /project/dash_agir/matthew.kutugata
#
# On SUNNY and other regular servers, normal home-directory behavior is used,
# and the virtual environment is created inside the repository.
#
# Usage:
#   bash setup.sh
#   TORCH_CUDA=cu124 bash setup.sh
#   PYTHON_VERSION=3.11 bash setup.sh
#
# Optional overrides:
#   FORCE_PLATFORM=scinet bash setup.sh
#   FORCE_PLATFORM=regular bash setup.sh
#   STORAGE_ROOT=/another/path bash setup.sh
#   VENV_DIR=/another/path/env bash setup.sh
# ==============================================================================

set -Eeuo pipefail
umask 002

# ------------------------------------------------------------------------------
# 1. General configuration
# ------------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-auto}"

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_DIR}/requirements.txt}"
LOCK_FILE="${LOCK_FILE:-${REPO_DIR}/uv.lock.txt}"

ORIGINAL_HOME="${HOME}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
HOST_FULL="$(hostname -f 2>/dev/null || hostname)"

# Optional override:
#   auto    = detect from hostname
#   scinet  = force SCINet behavior
#   regular = force normal server behavior
FORCE_PLATFORM="${FORCE_PLATFORM:-auto}"

# ------------------------------------------------------------------------------
# 2. Logging and error handling
# ------------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

on_exit() {
    local exit_code=$?

    if [[ "${exit_code}" -ne 0 ]]; then
        echo
        echo "=================================================="
        echo "Setup failed with exit code ${exit_code}."
        echo "Host: ${HOST_FULL}"
        echo "Platform mode: ${PLATFORM_MODE:-unknown}"
        echo
        echo "Common fixes:"
        echo "  - Check that the selected storage location is writable."
        echo "  - Run on a GPU node if CUDA detection is required."
        echo "  - Force CUDA with: TORCH_CUDA=cu124 bash setup.sh"
        echo "  - Override detection with FORCE_PLATFORM=scinet or regular."
        echo "=================================================="
    fi
}

trap on_exit EXIT

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "Required command not found: $1"
}

# ------------------------------------------------------------------------------
# 3. Platform detection
# ------------------------------------------------------------------------------

detect_platform() {
    case "${FORCE_PLATFORM,,}" in
        scinet)
            echo "scinet"
            return
            ;;
        regular)
            echo "regular"
            return
            ;;
        auto)
            ;;
        *)
            die \
                "Invalid FORCE_PLATFORM='${FORCE_PLATFORM}'. " \
                "Use auto, scinet, or regular."
            ;;
    esac

    local host_string
    host_string="${HOST_SHORT,,} ${HOST_FULL,,}"

    # Match common CERES and ATLAS naming forms, including compute nodes such as:
    #   ceres
    #   ceres.scinet.usda.gov
    #   ceres19-compute-0-1
    #   atlas
    #   atlas-login
    #   atlas123
    if [[ "${host_string}" =~ (^|[[:space:]._-])ceres([[:space:]._-]|[0-9]|$) ]] ||
       [[ "${host_string}" =~ (^|[[:space:]._-])atlas([[:space:]._-]|[0-9]|$) ]]; then
        echo "scinet"
    else
        echo "regular"
    fi
}

PLATFORM_MODE="$(detect_platform)"

# ------------------------------------------------------------------------------
# 4. Host-specific storage configuration
# ------------------------------------------------------------------------------

configure_storage() {
    if [[ "${PLATFORM_MODE}" == "scinet" ]]; then
        STORAGE_ROOT="${STORAGE_ROOT:-/project/dash_agir/matthew.kutugata}"
        RUNTIME_HOME="${RUNTIME_HOME:-${STORAGE_ROOT}/.runtime_home}"
	VENV_DIR="${VENV_DIR:-${REPO_DIR}/.field_segmentation}"

        # Replace HOME during setup and activation so libraries that ignore XDG
        # variables still stay out of the user's limited SCINet home directory.
        export HOME="${RUNTIME_HOME}"

        export XDG_CACHE_HOME="${STORAGE_ROOT}/cache"
        export XDG_CONFIG_HOME="${STORAGE_ROOT}/config"
        export XDG_DATA_HOME="${STORAGE_ROOT}/data"
        export XDG_STATE_HOME="${STORAGE_ROOT}/state"

        export TMPDIR="${STORAGE_ROOT}/tmp"
        export TEMP="${TMPDIR}"
        export TMP="${TMPDIR}"

        export UV_INSTALL_DIR="${STORAGE_ROOT}/bin"
        export UV_CACHE_DIR="${STORAGE_ROOT}/cache/uv"
        export UV_PYTHON_INSTALL_DIR="${STORAGE_ROOT}/python"
        export UV_TOOL_DIR="${STORAGE_ROOT}/uv-tools"
        export UV_TOOL_BIN_DIR="${STORAGE_ROOT}/bin"

        export PIP_CACHE_DIR="${STORAGE_ROOT}/cache/pip"
        export PYTHONPYCACHEPREFIX="${STORAGE_ROOT}/cache/python/pycache"
        export MPLCONFIGDIR="${STORAGE_ROOT}/config/matplotlib"
        export WANDB_DIR="${STORAGE_ROOT}/wandb"
        export WANDB_CACHE_DIR="${STORAGE_ROOT}/cache/wandb"
        export TORCH_HOME="${STORAGE_ROOT}/cache/torch"
        export HF_HOME="${STORAGE_ROOT}/cache/huggingface"
        export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
        export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
        export NUMBA_CACHE_DIR="${STORAGE_ROOT}/cache/numba"
        export YOLO_CONFIG_DIR="${STORAGE_ROOT}/config/ultralytics"

        export PATH="${STORAGE_ROOT}/bin:${PATH}"
    else
        # SUNNY and other regular servers use normal home-directory behavior.
        STORAGE_ROOT="${STORAGE_ROOT:-${ORIGINAL_HOME}}"
        RUNTIME_HOME="${ORIGINAL_HOME}"
        VENV_DIR="${VENV_DIR:-${REPO_DIR}/.field_segmentation}"

        export HOME="${ORIGINAL_HOME}"

        # Do not override normal XDG, temp, or package cache locations here.
        # uv will use the user's standard ~/.cache, ~/.local, and related paths.

        UV_INSTALL_DIR="${UV_INSTALL_DIR:-${HOME}/.local/bin}"
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    fi
}

configure_storage

# ------------------------------------------------------------------------------
# 5. Directory creation and validation
# ------------------------------------------------------------------------------

prepare_storage() {
    if [[ "${PLATFORM_MODE}" == "scinet" ]]; then
        [[ -d "/project/dash_agir" ]] ||
            die \
                "Detected SCINet, but /project/dash_agir does not exist. " \
                "Host detection may be incorrect."

        mkdir -p \
            "${RUNTIME_HOME}" \
            "${XDG_CACHE_HOME}" \
            "${XDG_CONFIG_HOME}" \
            "${XDG_DATA_HOME}" \
            "${XDG_STATE_HOME}" \
            "${TMPDIR}" \
            "${UV_INSTALL_DIR}" \
            "${UV_CACHE_DIR}" \
            "${UV_PYTHON_INSTALL_DIR}" \
            "${UV_TOOL_DIR}" \
            "${PIP_CACHE_DIR}" \
            "${PYTHONPYCACHEPREFIX}" \
            "${MPLCONFIGDIR}" \
            "${WANDB_DIR}" \
            "${WANDB_CACHE_DIR}" \
            "${TORCH_HOME}" \
            "${HF_HOME}" \
            "${HUGGINGFACE_HUB_CACHE}" \
            "${TRANSFORMERS_CACHE}" \
            "${NUMBA_CACHE_DIR}" \
            "${YOLO_CONFIG_DIR}" \
            "$(dirname "${VENV_DIR}")"
    else
        mkdir -p \
            "${UV_INSTALL_DIR}" \
            "$(dirname "${VENV_DIR}")"
    fi

    local test_directory
    test_directory="$(dirname "${VENV_DIR}")"

    [[ -w "${test_directory}" ]] ||
        die "Environment parent directory is not writable: ${test_directory}"

    local test_file="${test_directory}/.setup_write_test_$$"
    touch "${test_file}" ||
        die "Unable to write to: ${test_directory}"
    rm -f "${test_file}"
}

print_configuration() {
    log "Detected host configuration:"
    echo "  Short hostname:      ${HOST_SHORT}"
    echo "  Full hostname:       ${HOST_FULL}"
    echo "  Platform mode:       ${PLATFORM_MODE}"
    echo "  Original home:       ${ORIGINAL_HOME}"
    echo "  Active HOME:         ${HOME}"
    echo "  Storage root:        ${STORAGE_ROOT}"
    echo "  Virtual environment: ${VENV_DIR}"

    if [[ "${PLATFORM_MODE}" == "scinet" ]]; then
        echo "  uv install dir:      ${UV_INSTALL_DIR}"
        echo "  uv cache:            ${UV_CACHE_DIR}"
        echo "  Python installs:     ${UV_PYTHON_INSTALL_DIR}"
        echo "  Temporary files:     ${TMPDIR}"
    else
        echo "  Storage behavior:    normal user home directories"
    fi
}

# ------------------------------------------------------------------------------
# 6. uv installation
# ------------------------------------------------------------------------------

install_uv_if_missing() {
    if command -v uv >/dev/null 2>&1; then
        log "uv found: $(command -v uv)"
        log "uv version: $(uv --version)"
        return
    fi

    log "uv not found. Installing uv..."

    if [[ "${PLATFORM_MODE}" == "scinet" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh |
            env \
                HOME="${HOME}" \
                XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
                XDG_DATA_HOME="${XDG_DATA_HOME}" \
                UV_INSTALL_DIR="${UV_INSTALL_DIR}" \
                sh
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    hash -r

    command -v uv >/dev/null 2>&1 ||
        die "uv installation failed"

    log "uv installed: $(command -v uv)"
    log "uv version: $(uv --version)"
}

verify_uv_storage() {
    if [[ "${PLATFORM_MODE}" != "scinet" ]]; then
        return
    fi

    local reported_cache
    reported_cache="$(uv cache dir)"

    [[ "${reported_cache}" == "${UV_CACHE_DIR}" ]] ||
        die \
            "uv cache is using '${reported_cache}', but expected " \
            "'${UV_CACHE_DIR}'."

    log "Confirmed uv cache directory: ${reported_cache}"
}

# ------------------------------------------------------------------------------
# 7. Requirements
# ------------------------------------------------------------------------------

create_requirements_if_missing() {
    if [[ -f "${REQUIREMENTS_FILE}" ]]; then
        log "Using existing requirements file: ${REQUIREMENTS_FILE}"
        return
    fi

    log "Creating requirements file: ${REQUIREMENTS_FILE}"

    cat > "${REQUIREMENTS_FILE}" <<'EOF'
# Core data and math
numpy
pandas

# Vision and plotting
matplotlib
pillow
scikit-image
albumentations
opencv-python

# Utilities
tqdm
pyyaml
gputil

# Configuration and tracking
hydra-core
omegaconf
wandb
fiftyone

# Segmentation and metrics
pytorch-lightning
segmentation-models-pytorch
torchmetrics

# Additional tools
ultralytics
ollama
EOF
}

# ------------------------------------------------------------------------------
# 8. CUDA detection
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 9. Activation helper
# ------------------------------------------------------------------------------

create_activation_script() {
    local activation_script="${REPO_DIR}/activate.sh"

    log "Writing activation helper: ${activation_script}"

    cat > "${activation_script}" <<'ACTIVATE_HEADER'
#!/usr/bin/env bash

_activation_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_activation_host_short="$(hostname -s 2>/dev/null || hostname)"
_activation_host_full="$(hostname -f 2>/dev/null || hostname)"
_activation_host_string="${_activation_host_short,,} ${_activation_host_full,,}"

if [[ "${_activation_host_string}" =~ (^|[[:space:]._-])ceres([[:space:]._-]|[0-9]|$) ]] ||
   [[ "${_activation_host_string}" =~ (^|[[:space:]._-])atlas([[:space:]._-]|[0-9]|$) ]]; then

    export FIELD_SEGMENTATION_PLATFORM="scinet"
    export FIELD_SEGMENTATION_STORAGE_ROOT="/project/dash_agir/matthew.kutugata"

    export HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/.runtime_home"

    export XDG_CACHE_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache"
    export XDG_CONFIG_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/config"
    export XDG_DATA_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/data"
    export XDG_STATE_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/state"

    export TMPDIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/tmp"
    export TEMP="${TMPDIR}"
    export TMP="${TMPDIR}"

    export UV_INSTALL_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/bin"
    export UV_CACHE_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/uv"
    export UV_PYTHON_INSTALL_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/python"
    export UV_TOOL_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/uv-tools"
    export UV_TOOL_BIN_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/bin"

    export PIP_CACHE_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/pip"
    export PYTHONPYCACHEPREFIX="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/python/pycache"
    export MPLCONFIGDIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/config/matplotlib"
    export WANDB_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/wandb"
    export WANDB_CACHE_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/wandb"
    export TORCH_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/torch"
    export HF_HOME="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/huggingface"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export NUMBA_CACHE_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/cache/numba"
    export YOLO_CONFIG_DIR="${FIELD_SEGMENTATION_STORAGE_ROOT}/config/ultralytics"

    export PATH="${FIELD_SEGMENTATION_STORAGE_ROOT}/bin:${PATH}"

    _activation_venv="${_activation_repo_dir}/.field_segmentation"
else
    export FIELD_SEGMENTATION_PLATFORM="regular"
    export FIELD_SEGMENTATION_STORAGE_ROOT="${HOME}"

    _activation_venv="${_activation_repo_dir}/.field_segmentation"
fi

if [[ ! -f "${_activation_venv}/bin/activate" ]]; then
    echo "Environment does not exist on this host:"
    echo "  ${_activation_venv}"
    echo
    echo "Run:"
    echo "  bash ${_activation_repo_dir}/setup.sh"
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_activation_venv}/bin/activate"

echo "Environment activated"
echo "  Platform: ${FIELD_SEGMENTATION_PLATFORM}"
echo "  Host:     ${_activation_host_full}"
echo "  Python:   $(command -v python)"

if [[ "${FIELD_SEGMENTATION_PLATFORM}" == "scinet" ]]; then
    echo "  HOME:     ${HOME}"
    echo "  Cache:    ${UV_CACHE_DIR}"
fi

unset _activation_repo_dir
unset _activation_host_short
unset _activation_host_full
unset _activation_host_string
unset _activation_venv
ACTIVATE_HEADER

    chmod +x "${activation_script}"
}

# ------------------------------------------------------------------------------
# 10. Main
# ------------------------------------------------------------------------------

main() {
    log "Starting field-segmentation environment setup"

    require_command curl

    prepare_storage
    print_configuration
    install_uv_if_missing
    verify_uv_storage
    create_requirements_if_missing

    log "Creating virtual environment: ${VENV_DIR}"

    uv venv \
        --python "${PYTHON_VERSION}" \
        --clear \
        "${VENV_DIR}"

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    log "Python executable: $(command -v python)"
    log "Python version: $(python --version)"

    log "Installing PyTorch with automatic backend selection"

    uv pip install \
        torch \
        torchvision \
	--torch-backend=auto

    log "Installing remaining dependencies"

    uv pip install \
        --index-strategy unsafe-best-match \
        -r "${REQUIREMENTS_FILE}"

    log "Verifying installation"

    python - <<'PYTHON'
import os
from pathlib import Path

import pytorch_lightning
import segmentation_models_pytorch
import torch
import ultralytics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print(f"Python home: {Path.home()}")
print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV')}")
print("Basic imports passed successfully.")
PYTHON

    log "Writing dependency snapshot: ${LOCK_FILE}"
    uv pip freeze > "${LOCK_FILE}"

    create_activation_script

    cat <<EOF

==================================================
Setup completed successfully
==================================================

Host:
  ${HOST_FULL}

Detected platform:
  ${PLATFORM_MODE}

Environment:
  ${VENV_DIR}

Activate it with:
  source ${REPO_DIR}/activate.sh
EOF
}

main "$@"

