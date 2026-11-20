"""Native PyTorch SDPA implementation of Cosmos two-way attention (CP=1).

Each sample is evaluated separately: AR reads its causal prefix; all DM tokens
read that prefix and each other. No sample or future-to-AR leakage is possible.
This explicit backend supports Ampere without the framework's custom wheels.
"""

import torch
import torch.nn.functional as F


def _sdpa(q, k, v, *, causal=False):
    return (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=causal,
            dropout_p=0.0,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        .squeeze(0)
        .transpose(0, 1)
        .flatten(-2)
    )


def gui_two_way_sdpa(
    packed_query_states,
    packed_key_states,
    packed_value_states,
    attention_mask,
    natten_metadata=None,
    memory_value=None,
    packed_key_states_normalized=None,
):
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        from_mode_splits,
        get_causal_seq,
        get_full_only_seq,
    )

    if memory_value is not None or natten_metadata is not None:
        raise ValueError("GUI SDPA does not support memory or sparse attention")
    if (
        attention_mask.is_three_way
        or getattr(attention_mask, "control_stream_token_ranges", None)
        or getattr(attention_mask, "flex_block_mask", None) is not None
        or getattr(attention_mask, "multiview_dense", None) is not None
        or packed_query_states.get("is_sharded", False)
    ):
        raise ValueError("GUI SDPA supports only ordinary two-way, unsharded token sequences")
    q_u, u_offsets = get_causal_seq(packed_query_states)
    q_g, g_offsets = get_full_only_seq(packed_query_states)
    k_u, _ = get_causal_seq(packed_key_states)
    v_u, _ = get_causal_seq(packed_value_states)
    norm_k = packed_key_states if packed_key_states_normalized is None else packed_key_states_normalized
    k_gu, _ = get_causal_seq(norm_k)
    k_gg, _ = get_full_only_seq(norm_k)
    v_g, _ = get_full_only_seq(packed_value_states)
    u_offsets, g_offsets = u_offsets.tolist(), g_offsets.tolist()
    if len(u_offsets) != len(g_offsets):
        raise ValueError("AR/DM sample boundaries differ")
    outputs_u, outputs_g = [], []
    for u0, u1, g0, g1 in zip(u_offsets[:-1], u_offsets[1:], g_offsets[:-1], g_offsets[1:], strict=True):
        if u1 <= u0 or g1 <= g0:
            raise ValueError("GUI SDPA requires nonempty AR and DM for every sample")
        outputs_u.append(_sdpa(q_u[u0:u1], k_u[u0:u1], v_u[u0:u1], causal=True))
        outputs_g.append(
            _sdpa(
                q_g[g0:g1],
                torch.cat([k_gu[u0:u1], k_gg[g0:g1]]),
                torch.cat([v_u[u0:u1], v_g[g0:g1]]),
            )
        )
    return from_mode_splits(torch.cat(outputs_u), torch.cat(outputs_g), packed_query_states), None


def install_gui_attention(network):
    for layer in network.language_model.model.layers:
        layer.self_attn.dispatch_attention_fn = gui_two_way_sdpa
