#!/usr/bin/env bash
set -euo pipefail

# Create or update a Python virtual environment named "calico-solver"
# and install/upgrade packages from requirements files.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_NAME="calico-solver"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using Python: ${PYTHON_BIN}"
echo "Virtualenv: ${VENV_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists. Reusing it."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
# Force standard PyPI and ignore any extra indexes from global/user config
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL || true
unset PIP_TRUSTED_HOST || true
# Ignore global/user/site pip config that might set custom indexes
export PIP_CONFIG_FILE="/dev/null"

python -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}" --isolated

REQ_MAIN="${PROJECT_ROOT}/requirements.in"
REQ_DEV="${PROJECT_ROOT}/requirements-dev.in"

# Install pip-tools first if we have .in files
if [ -f "${REQ_MAIN}" ] || [ -f "${REQ_DEV}" ]; then
    echo "Installing pip-tools..."
    pip install pip-tools --index-url "${PIP_INDEX_URL}" --isolated
fi

# Create a temporary combined requirements file for syncing
TEMP_REQ="${PROJECT_ROOT}/.temp-combined-requirements.txt"
trap "rm -f '${TEMP_REQ}'" EXIT

# Combine all requirements into one file
: > "${TEMP_REQ}"

if [ -f "${REQ_MAIN}" ]; then
    echo "# Main requirements" >> "${TEMP_REQ}"
    cat "${REQ_MAIN}" >> "${TEMP_REQ}"
    echo "" >> "${TEMP_REQ}"
fi

if [ -f "${REQ_DEV}" ]; then
    echo "# Dev requirements" >> "${TEMP_REQ}"
    # Skip constraint lines when combining
    grep -v '^-c\s' "${REQ_DEV}" >> "${TEMP_REQ}" || true
fi

if [ -s "${TEMP_REQ}" ]; then
    echo "Installing/syncing packages (will remove extras)..."
    pip-sync "${TEMP_REQ}" --pip-args "--index-url ${PIP_INDEX_URL} --isolated"
else
    echo "No requirements files found to install."
fi

# Install and configure pre-commit hooks
PRE_COMMIT_CONFIG="${PROJECT_ROOT}/.pre-commit-config.yaml"
if [ -f "${PRE_COMMIT_CONFIG}" ]; then
    echo "Setting up pre-commit hooks..."
    # Clean any existing pre-commit cache to avoid repo issues
    pre-commit clean || true
    # Install with the same PyPI-only environment
    PIP_INDEX_URL="${PIP_INDEX_URL}" PIP_CONFIG_FILE="${PIP_CONFIG_FILE}" pre-commit install --overwrite
    echo "Pre-commit hooks installed. Ruff will now check your code on every commit."
else
    echo "No .pre-commit-config.yaml found. Skipping pre-commit setup."
fi

echo "Virtual environment ready. Python executable: $(command -v python)"
echo "To activate this environment in your shell:"
echo "  source '${VENV_DIR}/bin/activate'"
