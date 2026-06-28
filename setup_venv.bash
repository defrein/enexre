#!/usr/bin/env bash
set -euo pipefail

KERNEL_NAME="enexre"
KERNEL_DISPLAY_NAME="Python (enexre)"
VENV_DIR=".venv"

cd "$(dirname "$0")"

echo "Creating virtual environment in ${VENV_DIR}..."
python -m venv "${VENV_DIR}"

if [[ -f "${VENV_DIR}/Scripts/python.exe" ]]; then
  PYTHON="${VENV_DIR}/Scripts/python.exe"
else
  PYTHON="${VENV_DIR}/bin/python"
fi

echo "Upgrading pip..."
"${PYTHON}" -m pip install --upgrade pip

echo "Installing notebook and experiment dependencies from requirements.txt..."
"${PYTHON}" -m pip install -r requirements.txt

echo "Registering Jupyter kernel: ${KERNEL_DISPLAY_NAME}"
"${PYTHON}" -m ipykernel install \
  --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "Done."
echo "Open Protocol.ipynb in VS Code and select kernel: ${KERNEL_DISPLAY_NAME}"
