#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${GUI_BACKBONE_PATH:?Set the GUI-Libra backbone path}"
export GUI_TRAIN_CONFIG=${GUI_TRAIN_CONFIG:-"$repo/examples/gui_mot/configs/gui_libra_joint_h1.toml"}
export GUI_INSTRUCTION_LEVEL=${GUI_INSTRUCTION_LEVEL:-low}
export GUI_RUN_NAME=${GUI_RUN_NAME:-"gui_libra_joint_h1_${GUI_INSTRUCTION_LEVEL}_smoke"}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"$repo/outputs/train"}
case "$GUI_INSTRUCTION_LEVEL" in
    high|low) ;;
    *) echo 'GUI_INSTRUCTION_LEVEL must be high or low' >&2; exit 2 ;;
esac
exec bash "$repo/examples/gui_mot/scripts/train_gui_joint.sh" "$@"
