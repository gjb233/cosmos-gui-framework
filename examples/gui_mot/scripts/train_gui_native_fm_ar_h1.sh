#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -n ${MOT_CHECKPOINT_PATH:-} ]]; then
    echo 'Native FM-AR training must start from the Cosmos base checkpoint' >&2
    exit 2
fi

export GUI_HORIZON=1
export GUI_INSTRUCTION_LEVEL=low
export GUI_MOT_JOINT=1
export GUI_HYBRID_AR=0
export GUI_HYBRID_KD_WEIGHT=0
export GUI_NORMALIZE_PLAN=0
export GUI_GROUPED_CE=${GUI_GROUPED_CE:-1}
# The standalone AR CE pass needs its own activation recomputation.
export GUI_AR_ACTIVATION_CHECKPOINT=${GUI_AR_ACTIVATION_CHECKPOINT:-1}
unset GUI_ACTION_PLAN_CACHE
export GUI_TRAIN_CONFIG=${GUI_TRAIN_CONFIG:-"$repo/examples/gui_mot/configs/gui_native_fm_ar_train_h1.toml"}
export GUI_RUN_NAME=${GUI_RUN_NAME:-gui_native_fm_ar_h1_v1}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"$repo/outputs/train-native-fm-ar-h1"}

: "${GUI_TRAIN_MANIFEST:?Set the train manifest with official eval episodes removed}"
: "${BASE_CHECKPOINT_PATH:?Set the original Cosmos base checkpoint}"
: "${GUI_BACKBONE_PATH:?Set the GUI-Libra backbone}"

exec bash "$repo/examples/gui_mot/scripts/train_gui_libra_joint.sh" "$@"
