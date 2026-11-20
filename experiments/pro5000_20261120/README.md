# Pro5000 training and evaluation snapshots (2026-11-20)

This snapshot preserves the scripts and static configurations used on Pro5000 after the 2026-11-11 handoff. It contains GUI-Libra CE+FM and CE-only experiments, MAI-UI native VLM CE training, offline/AndroidWorld evaluation, prompt comparisons, and FM visualization repairs.

## Entry points

- `training/mai-ui-vlm-ce-20261119/train.py`: native Transformers + PEFT + DeepSpeed ZeRO-2 VLM-only training; launcher and exact config are alongside it.
- `training/gui-libra-awhigh-aclow-ceonly-20261118/`: joint-runtime CE-only ablation.
- `training/gui-libra-awhigh-aclow-fm2-20261117/`: CE+FM weight-2 configuration.
- `step200-suite-20261118/`: evaluation and repaired FM visualization scripts.
- `scripts/`: preparation, filtering, evaluation, and environment helpers.

## Runtime assumptions

These are historical executable snapshots, not portable launchers. Review scripts before execution: absolute `/data/mai` paths, eight GPUs, ports, checkpoint names, process controllers and original environment assumptions are retained. Copy the selected snapshot to its original path or adapt paths explicitly. Do not run all historical launchers or queues. Model weights, manifests, ordered sampling schedules, images, generated results, optimizer state and credentials are external assets, not Git contents.

The latest two CE runs completed 725 steps with global batch 256, microbatch 8 and accumulation 4. MAI-UI uses sample-mean answer-only CE including EOS, rank-strided draws, original-resolution images and attention+MLP LoRA rank16/alpha32. Its PEFT initialization and native implementation differ from the joint runtime; it is not a bitwise-equivalent reproduction. GUI-Libra CE keeps the joint graph with FM weight zero.

## Provenance and validation

`source-manifest.json` records hashes of the imported Pro5000 files. The `experiments/mai_20261111` tree is an earlier handoff snapshot, and its README records historical validation, not a new validation claim for this commit. Original exported source label: f2f8996. That commit's Git objects were not present on the training server; this branch is based on the existing `gui-cosmos-fm-action` branch and imports its source changes without rewriting remote history. Existing Android GUI domain registration is retained.

All imported Python files were checked with AST parsing, and the publish set was scanned for common credential literals. Local CPU regression execution is limited by older PyTorch and missing webdataset; all 24 CPU regression tests passed in the matching server environment on 2026-11-20. Archived scripts retain their original formatting, including existing whitespace warnings. No GPU experiments were rerun for this source publication.
