import torch

def prepare_native_ar(network, packed_seq, last_hidden_state):
    """Teacher-forced native AR logits with a protected joint-state adapter.

    This runs inside ``Cosmos3VFMNetwork.forward`` so FSDP parameters remain
    materialized. Clean future/action targets are never used as AR features;
    context comes from the joint denoiser hidden states.
    """
    if not hasattr(packed_seq, "gui_action_target_ids"):
        return
    stage = getattr(packed_seq, "gui_hybrid_stage", None)
    if not getattr(packed_seq, "gui_mot_joint", False) and not stage.enable_ar_loss:
        return
    targets = packed_seq.gui_action_target_ids
    payload = packed_seq.gui_multimodal_inputs
    if len(targets) != 1 or len(payload) != 1:
        raise ValueError("Initial hybrid AR path requires per-rank batch size one")
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
        prepare_multimodal_reasoner_inputs,
    )

    sample = {
        key: value.to(last_hidden_state.device)
        for key, value in payload[0].items()
        if key in ("input_ids", "pixel_values", "image_grid_thw", "attention_mask")
    }
    with torch.no_grad(), torch.autocast(device_type=last_hidden_state.device.type, enabled=False):
        prompt, visual_mask, deepstack, position_ids, _ = prepare_multimodal_reasoner_inputs(
            network.language_model, **sample
        )
    labels = torch.tensor(targets[0], device=prompt.device, dtype=torch.long)
    token_masks = getattr(packed_seq, "gui_action_target_masks", None)
    token_mask = (
        torch.ones_like(labels, dtype=torch.bool)
        if token_masks is None
        else torch.tensor(token_masks[0], device=prompt.device, dtype=torch.bool)
    )
    if token_mask.shape != labels.shape or not token_mask.any():
        raise ValueError("Hybrid AR answer token mask is invalid")
    action_inputs = labels[:-1]
    action_embeddings = network.language_model.model.embed_tokens(action_inputs.unsqueeze(0))
    compute_dtype = (
        last_hidden_state.dtype if last_hidden_state.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    )
    prompt = prompt.to(dtype=compute_dtype)
    action_embeddings = action_embeddings.to(dtype=compute_dtype)
    deepstack = [item.to(dtype=compute_dtype) for item in deepstack]
    inputs = torch.cat([prompt, action_embeddings], dim=1)
    extra = action_inputs.numel()
    if extra:
        visual_mask = torch.cat([visual_mask, visual_mask.new_zeros((1, extra))], dim=1)
        tail = position_ids[:, :, -1:] + torch.arange(1, extra + 1, device=position_ids.device)
        position_ids = torch.cat([position_ids, tail], dim=2)
    if not getattr(packed_seq, "gui_native_fm_ar", False):
        raise ValueError("Pipeline frontend supports native FM/AR only")
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        cos, sin = network.language_model.model.rotary_emb(inputs, position_ids=position_ids)
    return dict(ar_hidden=inputs, ar_cos=cos, ar_sin=sin,
                ar_visual_mask=visual_mask, ar_deepstack=deepstack,
                labels=labels, token_mask=token_mask, prompt_length=prompt.shape[1],
                weights=getattr(packed_seq, "gui_action_target_weights", None))

def prepare_fm_decoder(model, pack, position_ids):
    from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, from_all_seq
    meta = get_gen_seq(pack)
    cos, sin = model.rotary_emb(meta, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1))
    position_embeddings = (from_all_seq(cos.squeeze(0), pack), from_all_seq(sin.squeeze(0), pack))
    pack = dict(pack)
    visual = pack.pop("_gui_visual", None)
    return pack, position_embeddings, visual

def finish_native_ar(network, ar_hidden, preparation):
    hidden = network.language_model.model.norm(ar_hidden)
    labels = preparation["labels"]
    start = preparation["prompt_length"] - 1
    action_hidden = hidden[:, start:start+labels.numel()]
    with torch.autocast(device_type=action_hidden.device.type, dtype=action_hidden.dtype):
        logits = network.language_model.lm_head(action_hidden)
    result = dict(gui_ar_logits=logits, gui_ar_labels=labels.unsqueeze(0),
                  gui_ar_token_mask=preparation["token_mask"].unsqueeze(0))
    if preparation["weights"] is not None:
        result["gui_ar_token_weights"] = torch.tensor(preparation["weights"],device=labels.device,dtype=torch.float32)
    return result

def finish_fm_decoder(model, pack):
    from cosmos_framework.data.generator.sequence_packing.runtime import zeros_like,get_und_seq,get_gen_seq,set_und_seq,set_gen_seq
    result = zeros_like(pack)
    set_und_seq(result,model.norm(get_und_seq(pack)))
    set_gen_seq(result,model.norm_moe_gen(get_gen_seq(pack)))
    return result
