# MAI GUI training and benchmark code snapshot — 2026-11-11

This branch captures mixed GUI CE and CE+FM experiments. The active training
source is synchronized into the normal cosmos_framework tree. The original
repository's five uncommitted edits are preserved in the preceding commit;
this commit records the actual runtime source, including objective differences.
Live training directories were not modified.

## Layout

| Directory | Contents |
| --- | --- |
| data_preparation | Eight-source preparation and action/image/split audits |
| recipes/mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem | FSDP CE+FM continuation and migration |
| recipes/mai-ui-70283-resume350-1f1b7-micro4-gb56-a22 | PyTorch 1F1B and checkpoint verification |
| recipes/dual-rank-resume | ms-swift pure CE controller and metrics |
| recipes/mai-input-length-audit-70933-20261109 | Length audit and historical reservation helper |
| pipeline | Shared stages, activation transport, metadata and endpoints |
| native_endpoint, native_template | Native frontend and target dependencies |
| benchmark | Framework/microbatch snapshots and historical controllers |
| evaluation | AndroidControl and MM-Mind2Web queues and clients |
| fm_visualization | Test-sample FM prediction and comparison gallery |

## External assets and execution

These are provenance-preserving snapshots, not relocatable turnkey launchers.
Scripts retain original absolute cluster paths and host/GPU assumptions.
Read and adapt each launcher before running it. This directory alone does not
provision the runtime.

Required external assets include original model/VAE weights, the filtered
70283-record manifest and ordered schedules, images, native endpoint/stage
reference exports, migration-350.pt and matching Python environments.
Stage reference exports use trusted Python pickles: load only project-created
artifacts. Data, checkpoints, credentials, environments, predictions, generated
images and logs are excluded.

source-manifest.json maps imported files to original paths and SHA-256 hashes.
Static config.json files retain historical settings; effective settings can
be overridden by drivers. Do not infer actual batch size from producer config.

## Benchmark protocol and current status

a22 GPUs 0–6, CE+FM, checkpoint/Adam/RNG step 350, draws 20800–21023,
global batch 56. Step 351 warms up; steps 352–354 measure 168 samples.
Use the slowest rank's synchronized end-to-end time including preparation.
Startup and checkpoint saving are separate. Verify exact IDs/instruction
levels, 584 gradients and updates, seven model/Adam/scheduler shards,
288 AR adapter tensors and producer RNG.

Short-run throughput in samples/hour, rounded:
FSDP micro1/2/4/8: 486/497/491/488;
1F1B micro1/4: 277/323; GPipe micro1: 257.
These measurements do not establish a final optimum.
1F1B micro8 OOMed before its first update.
Early pipeline attempts failed from a Python queue-module name collision;
micro2 also had a row-count shape error. Corrected source is included.
Failed starts are not throughput results. DeepSpeed adapters are prepared
but not runtime-validated. Serial micro4/global56 has completed its first
true update; final three-step throughput is pending.

The user stopped the previous matrix to test serial execution. Archived
grid/DeepSpeed controllers must not be restarted automatically. a23 project
evaluation is stopped. This snapshot does not change running jobs.

## Validation

Imported Python files pass AST parsing. These 24 CPU tests passed in the
existing project environment without using training GPUs:

    PYTHONPATH=tests:. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -m unittest gui_mixed_supervision_test gui_loss_integration_test gui_native_fm_ar_test gui_training_audit_test

Coverage includes mixed supervision, scalar CE/FM losses and gradients,
native targets, loader resume and recomputation. This is not a full rerun
of every archived experiment or the repository's complete test suite.
