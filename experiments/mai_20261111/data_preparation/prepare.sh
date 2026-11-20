#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m unittest -v test_prepare.py
"$PYTHON" prepare.py inventory --config mixture.json
if [[ "${EXTRACT_IMAGES:-0}" == 1 ]]; then
  "$PYTHON" extract_images.py /rscratch/wzzheng/datasets/gui-libra-81k-sft-20261028-v1
fi
if [[ -f original-evidence.jsonl && -f heldout.json ]]; then
  "$PYTHON" prepare.py build --evidence original-evidence.jsonl --heldout heldout.json
  "$PYTHON" prepare.py schedule --phase 1 --draws 81449
  "$PYTHON" prepare.py schedule --phase 2 --draws 162898
else
  printf '%s\n' 'Inventory complete. Original-annotation joins and heldout index are required before training.'
fi
