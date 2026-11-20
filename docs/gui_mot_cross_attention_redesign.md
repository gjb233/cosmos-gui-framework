# GUI MoT independent cross-attention redesign

The GUI MoT implementation and the required Cosmos framework changes live in
this repository. GUI-specific Python modules are under `cosmos_framework/gui_mot`;
the H1 training recipe and launch scripts are under `examples/gui_mot`.

The AR reasoner computes its native causal hidden states without adding FM keys
or values to native self-attention. After the last AR layer, a separate
cross-attention branch reads tokens derived from the predicted FM plan and
future. Its residual output is added to AR hidden states before the unchanged
language-model head. The branch's Q/K/V weights start normally, its output
projection `W_o` starts at exactly zero, and its gate starts at one. This makes
initial AR hidden states and logits exactly equal to the native path while
giving `W_o` a first-step gradient. Q/K/V and the FM conditioner begin learning
once `W_o` becomes nonzero. The output projection has no bias, so there is no
second nonzero path to the residual at initialization.

Fresh MoT training must load the original base weights, not a checkpoint from
the older bridge. The optimizer selection now includes both the cross-attention
branch and plan/future conditioner. Training logs include
`gui_ar_residual_norm_ratio` and `gui_ar_max_logit_delta` to track how strongly
the new branch changes AR hidden states and logits. Check these alongside AR
accuracy and the official AndroidControl evaluation before interpreting a
lower training loss as better action quality.

For H1 low, use `examples/gui_mot/scripts/train_gui_libra_mot_cross_attn_h1.sh` after setting
the original base checkpoint, GUI-Libra backbone, VAE, training manifest, and
native-JSON plan cache. The wrapper rejects `MOT_CHECKPOINT_PATH` so the old
MoT checkpoints cannot silently initialize this new branch.
