# GUI MoT H1 training

The GUI model code is in `cosmos_framework/gui_mot/`. The training recipe is
`examples/gui_mot/configs/gui_libra_mot_androidcontrol_train_h1.toml`.
Run from this repository; no second framework checkout or patch step is needed.

Set these paths to local assets before launching:

```bash
export GUI_PYTHON=/path/to/training-env/bin/python
export GUI_BACKBONE_PATH=/path/to/GUI-Libra-8B
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
export GUI_TRAIN_MANIFEST=/path/to/train-transitions.jsonl
export GUI_ACTION_PLAN_CACHE=/path/to/native-json-action-plans.json
```

Check configuration without starting GPU training:

```bash
bash examples/gui_mot/scripts/train_gui_libra_mot_cross_attn_h1.sh --dryrun
```

Launch with the allocated GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
bash examples/gui_mot/scripts/train_gui_libra_mot_cross_attn_h1.sh
```

The launcher requires a fresh Cosmos base checkpoint and rejects
`MOT_CHECKPOINT_PATH` so an older GUI MoT checkpoint cannot initialize this
zero-output cross-attention model. Set `GUI_TRAIN_CONFIG` to a different TOML
only when changing the schedule deliberately.

## Official AndroidControl evaluation

Use the MoT DCP evaluation callback for the official 398 Low samples. The
standalone `eval_gui_libra_official_androidcontrol.py` evaluates a plain
GUI-Libra model; it is not the entry point for a MoT DCP checkpoint.

Set the asset paths above, then set the official sample and screenshot paths:

```bash
export GUI_OFFICIAL_SAMPLES=/path/to/AndroidControl/data/500_steps_filtered.json
export GUI_OFFICIAL_SCREENSHOT_DIR=/path/to/AndroidControl_images
export GUI_TRAIN_CONFIG="$PWD/examples/gui_mot/configs/gui_libra_mot_official_eval_h1.toml"
export GUI_MOT_JOINT=1 GUI_HYBRID_AR=0 GUI_HORIZON=1
export GUI_TARGET_HEIGHT=2400 GUI_TARGET_WIDTH=1088
export GUI_OFFICIAL_MAX_NEW_TOKENS=512 GUI_OFFICIAL_SAMPLING_STEPS=20
export GUI_FSDP_MASTER_DTYPE=bfloat16 GUI_DTENSOR_SAFE_INIT=1
export GUI_REASONER_FORCE_TORCH_SDPA=1 GUI_VISION_NATIVE_CPU_ROPE=1
export GUI_VISION_DETERMINISTIC_CONV=1
export GUI_AR_NATIVE_NO_CACHE=1 GUI_AR_NATIVE_STOP_ONLY=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1
```

For **ckpt-0**, set `BASE_CHECKPOINT_PATH` to the fresh Cosmos base DCP and
leave `MOT_CHECKPOINT_PATH` unset. For an already trained MoT checkpoint, set
both paths to the same DCP directory (for example, `iter_000002500`) so the
learned MoT weights are restored rather than skipped:

```bash
# ckpt-0:
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
unset MOT_CHECKPOINT_PATH

# Or, for a trained checkpoint:
export BASE_CHECKPOINT_PATH=/path/to/iter_000002500
export MOT_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH"
```

Run one evaluation per GPU. Choose a new output directory for each run; the
callback exits after evaluation, before any training step:

```bash
export GUI_OFFICIAL_DCP_EVAL_OUTPUT="$PWD/outputs/official-low-ckpt2500"
export GUI_RUN_NAME=gui_mot_official_low_ckpt2500
bash examples/gui_mot/scripts/train_gui_libra_joint.sh
```

Predictions are written to
`$GUI_OFFICIAL_DCP_EVAL_OUTPUT/shard-0/predictions.jsonl`. Score them with the
official evaluator from its own directory:

```bash
cd /path/to/AndroidControl
"$GUI_PYTHON" eval_rl.py \
  --sample_file data/500_steps_filtered.json \
  --plan_file "$GUI_OFFICIAL_DCP_EVAL_OUTPUT/shard-0/predictions.jsonl" \
  --relative_coord 1
```

The evaluation still needs `GUI_TRAIN_MANIFEST` and `GUI_ACTION_PLAN_CACHE`
from the asset setup above because the model initialization builds the
training dataloader before the evaluation callback runs.
