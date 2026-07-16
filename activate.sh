#!/usr/bin/env bash

_activation_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_activation_host_short="$(hostname -s 2>/dev/null || hostname)"
_activation_host_full="$(hostname -f 2>/dev/null || hostname)"
_activation_host_string="${_activation_host_short,,} ${_activation_host_full,,}"

if [[ "${_activation_host_string}" =~ (^|[[:space:]._-])ceres([[:space:]._-]|[0-9]|$) ]] ||
   [[ "${_activation_host_string}" =~ (^|[[:space:]._-])atlas([[:space:]._-]|[0-9]|$) ]]; then

    export FIELD_SEGMENTATION_PLATFORM="scinet"
    export FIELD_SEGMENTATION_STORAGE_ROOT="/project/dash_agir/${USER:?USER is not set}"

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
