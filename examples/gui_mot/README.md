# GUI MoT H1 video-only training

The GUI model code is in `cosmos_framework/gui_mot/`. The training recipe is
`examples/gui_mot/configs/gui_libra_mot_androidcontrol_train_h1.toml`.
Run from this repository; no second framework checkout or patch step is needed.
This branch disables Cosmos continuous action generation. It predicts only the
next-frame latent and passes that prediction through zero-output cross-attention
to the GUI-Libra action-text head. The final action is still trained with CE.

Set these paths to local assets before launching:

```bash
export GUI_PYTHON=/path/to/training-env/bin/python
export GUI_BACKBONE_PATH=/path/to/GUI-Libra-8B
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
export GUI_TRAIN_MANIFEST=/path/to/train-transitions-with-eval-episodes-excluded.jsonl
```

Check configuration without starting GPU training:

```bash
bash examples/gui_mot/scripts/train_gui_libra_mot_video_only_h1.sh --dryrun
```

Launch with the allocated GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
bash examples/gui_mot/scripts/train_gui_libra_mot_video_only_h1.sh
```

The launcher requires a fresh Cosmos base checkpoint, defaults teacher KL to
zero, and ignores any action-plan cache. It rejects `MOT_CHECKPOINT_PATH` so
the older action-plus-video MoT checkpoint cannot silently initialize the
video-only architecture. Set `GUI_TRAIN_CONFIG` to a different TOML only when
changing the schedule deliberately.

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
export GUI_HYBRID_KD_WEIGHT=0
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
leave `MOT_CHECKPOINT_PATH` unset. For a video-only checkpoint from this branch,
set both paths to the same DCP directory so learned weights are restored:

```bash
# ckpt-0:
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
unset MOT_CHECKPOINT_PATH

# Or, for a trained video-only checkpoint:
export BASE_CHECKPOINT_PATH=/path/to/video-only/iter_000002000
export MOT_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH"
```

Run one evaluation per GPU. Choose a new output directory for each run; the
callback exits after evaluation, before any training step:

```bash
export GUI_OFFICIAL_DCP_EVAL_OUTPUT="$PWD/outputs/official-low-video-only-ckpt2000"
export GUI_RUN_NAME=gui_mot_video_only_official_low_ckpt2000
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

Evaluation still needs `GUI_TRAIN_MANIFEST` because model initialization builds
the training dataloader before the evaluation callback runs. No action-plan
cache is loaded on this branch. The earlier action-plus-video checkpoints are
not compatible with the video-only model definition.
