#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python=${GUI_PYTHON:-python3}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
if [[ ${1:-} == --dryrun ]]; then
    exec "$python" -m cosmos_framework.gui_mot.train_joint --sft-toml "${GUI_TRAIN_CONFIG:-$repo/examples/gui_mot/configs/gui_joint_h1.toml}" "$@"
fi
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE for the GPUs allocated to this training run}"
exec "$python" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m cosmos_framework.gui_mot.train_joint --sft-toml "${GUI_TRAIN_CONFIG:-$repo/examples/gui_mot/configs/gui_joint_h1.toml}" "$@"
