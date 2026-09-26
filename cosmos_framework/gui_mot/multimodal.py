"""Current-screen MAI/Qwen3-VL prefix for the Cosmos joint training forward."""

from contextlib import contextmanager

import torch


@contextmanager
def _without_lora(module):
    """Use the frozen pretrained AR weights as the KD teacher."""
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    layers = [layer for layer in module.modules() if isinstance(layer, LoraInjectedLinear)]
    alphas = [layer._lora_alpha for layer in layers]
    try:
        for layer in layers:
            layer._lora_alpha = 0.0
        yield
    finally:
        for layer, alpha in zip(layers, alphas, strict=True):
            layer._lora_alpha = alpha


def capture_current_screens(batch):
    """Snapshot current pixels before the native VAE normalizes video in place."""
    if "gui_current_screens" in batch:
        return
    screens = []
    for video in batch["video"]:
        while isinstance(video, list) and len(video) == 1:
            video = video[0]
        if not isinstance(video, torch.Tensor):
            raise TypeError("GUI prefix requires exactly one video item per sample")
        if video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        if video.ndim != 4 or video.shape[1] < 5 or (video.shape[1] - 1) % 4:
            raise ValueError("Expected one 1+4*horizon frame clip per sample")
        current = video[:, 0].detach().cpu()
        if current.dtype != torch.uint8:
            raise ValueError("Prefix requires original uint8 screen pixels")
        screens.append(current.clone())
    batch["gui_current_screens"] = screens


def prepare_prefix(processor, batch):
    """CPU processor sees only frame zero, even when the batch contains targets."""
    from PIL import Image

    capture_current_screens(batch)
    inputs = []
    screens = batch["gui_current_screens"]
    while isinstance(screens, list) and len(screens) == 1 and isinstance(screens[0], list):
        screens = screens[0]
    if isinstance(screens, torch.Tensor) and screens.ndim == 4:
        screens = list(screens)
    elif (
        isinstance(screens, list)
        and len(screens) == 1
        and isinstance(screens[0], torch.Tensor)
        and screens[0].ndim == 4
    ):
        screens = list(screens[0])
    systems = batch.get("gui_system_prompt")
    if systems is None:
        systems = [None] * len(batch["gui_current_screens"])
    while isinstance(systems, list) and len(systems) == 1 and isinstance(systems[0], list):
        systems = systems[0]
    for current, caption, system in zip(screens, batch["ai_caption"], systems, strict=True):
        # The native trainer moves metadata tensors to CUDA before prefix
        # preparation; PIL/AutoProcessor still require host pixels.
        image = Image.fromarray(current.detach().cpu().permute(1, 2, 0).numpy())
        messages = []
        if system:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": caption},
                ],
            }
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append(dict(processor(text=[text], images=[image], return_tensors="pt")))
    return inputs


def encode_prefix(network, packed_seq):
    """Called inside the FSDP network forward by the pinned framework patch."""
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
        prepare_multimodal_reasoner_inputs,
    )

    payload = packed_seq.gui_multimodal_inputs
    device = packed_seq.text_ids.device
    embeddings, masks, positions, stacks = [], [], [], []
    expected = []
    for sample in payload:
        sample = {
            k: v.to(device)
            for k, v in sample.items()
            if k in ("input_ids", "pixel_values", "image_grid_thw", "attention_mask")
        }
        special = packed_seq.gui_special_tokens
        ids = sample["input_ids"]
        prefix = [special["bos_token_id"]] if "bos_token_id" in special else []
        suffix = [special["eos_token_id"], special["start_of_generation"]]
        sample["input_ids"] = torch.cat([ids.new_tensor([prefix]), ids], dim=1)
        sample["attention_mask"] = torch.ones_like(sample["input_ids"])
        # All MAI parameters are frozen in stage one; retain DM gradients below.
        with torch.no_grad():
            emb, mask, deep, pos, _ = prepare_multimodal_reasoner_inputs(network.language_model, **sample)
            # Cosmos reuses <|vision_start|> as its generation separator. Passing
            # that dangling final token to Qwen's image mRoPE parser makes it
            # read beyond the sequence. Add the native suffix AFTER image parsing.
            tail_ids = ids.new_tensor([suffix])
            tail_emb = network.language_model.model.embed_tokens(tail_ids)
            emb = torch.cat([emb, tail_emb.to(emb)], dim=1)
            mask = torch.cat([mask, mask.new_zeros((1, len(suffix)))], dim=1)
            tail_pos = pos[:, :, -1:] + torch.arange(1, len(suffix) + 1, device=pos.device)
            pos = torch.cat([pos, tail_pos], dim=2)
        embeddings.append(emb.squeeze(0))
        masks.append(mask.squeeze(0))
        positions.append(pos.squeeze(1))
        stacks.append(deep)
        expected.append(torch.cat([sample["input_ids"], tail_ids], dim=1).flatten())
    if not torch.equal(torch.cat(expected), packed_seq.text_ids):
        raise ValueError("Packed prefix differs from current-screen processor tokens")
    text = torch.cat(embeddings)
    packed = text.new_zeros((packed_seq.sequence_length, network.hidden_size))
    packed[packed_seq.text_indexes] = text
    if packed_seq.position_ids.ndim != 2 or packed_seq.position_ids.shape[0] != 3:
        raise ValueError("MAI multimodal prefix requires three-axis mRoPE")
    packed_seq.position_ids = packed_seq.position_ids.clone()
    packed_seq.position_ids[:, packed_seq.text_indexes] = torch.cat(positions, dim=1).to(packed_seq.position_ids)
    packed_seq.gui_deepstack = [torch.cat([s[i] for s in stacks]) for i in range(len(stacks[0]))]
    packed_seq.gui_visual_mask = torch.cat(masks)
    return packed, text.dtype


def attach_deepstack(input_pack, packed_seq):
    if hasattr(packed_seq, "gui_visual_mask"):
        input_pack["_gui_visual"] = (
            packed_seq.gui_visual_mask,
            packed_seq.gui_deepstack,
        )


def add_deepstack(hidden_states, gui_visual, layer_index):
    if gui_visual is None or layer_index >= len(gui_visual[1]):
        return
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        get_und_seq,
        set_und_seq,
    )

    mask, stacks = gui_visual
    und = get_und_seq(hidden_states)
    if und.shape[0] < mask.numel():
        raise ValueError("GUI deepstack does not support sharded understanding tokens")
    if und.shape[0] > mask.numel():
        # Native two-way packing appends an isolated padding segment to both
        # towers. Padding is never an image placeholder or a DeepStack target.
        mask = torch.nn.functional.pad(mask, (0, und.shape[0] - mask.numel()), value=False)
    und = und.clone()
    und[mask] = und[mask] + stacks[layer_index].to(und)
    set_und_seq(hidden_states, und)


def hybrid_ar_forward(network, packed_seq, last_hidden_state, output_dict):
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
    with torch.no_grad():
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
    # DeepStack applies only to image placeholders in the unchanged prompt.
    with torch.autocast(device_type="cuda", dtype=compute_dtype):
        base = network.language_model.model.reasoner_forward(
            input_ids=None,
            cache=None,
            inputs_embeds=inputs,
            position_ids=position_ids,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
        )
    start = prompt.shape[1] - 1
    action_hidden = base[:, start : start + labels.numel()]
    mot_joint = bool(getattr(packed_seq, "gui_mot_joint", False))
    video_only = bool(getattr(packed_seq, "gui_video_only", False))
    if not video_only:
        if not hasattr(packed_seq, "gui_sigmas_action"):
            raise ValueError("Hybrid AR path is missing diffusion sigma metadata")
        noisy_plan = packed_seq.action.tokens[0]
        predicted_plan_velocity = output_dict["preds_action"][0]
        plan_sigma = packed_seq.gui_sigmas_action[0]
        while plan_sigma.ndim < noisy_plan.ndim:
            plan_sigma = plan_sigma.unsqueeze(-1)
        plan_x0 = noisy_plan.float() - plan_sigma.float() * predicted_plan_velocity.float()

    noisy_future = packed_seq.vision.tokens[0]
    predicted_future_velocity = output_dict["preds_vision"][0]
    if noisy_future.ndim == 5 and noisy_future.shape[0] == 1:
        noisy_future = noisy_future[0]
    if predicted_future_velocity.ndim == 5 and predicted_future_velocity.shape[0] == 1:
        predicted_future_velocity = predicted_future_velocity[0]
    future_sigma = packed_seq.gui_sigmas_vision[0]
    if noisy_future.ndim == 4 and future_sigma.ndim == 3 and future_sigma.shape[0] == noisy_future.shape[1]:
        future_sigma = future_sigma.unsqueeze(0)
    else:
        while future_sigma.ndim < noisy_future.ndim:
            future_sigma = future_sigma.unsqueeze(-1)
    future_x0 = noisy_future.float() - future_sigma.float() * predicted_future_velocity.float()
    if not mot_joint and stage.detach_joint:
        plan_x0 = plan_x0.detach()
        future_x0 = future_x0.detach()
    bridge_scale = 1.0 if mot_joint else stage.bridge_scale
    with torch.autocast(device_type="cuda", dtype=compute_dtype):
        joint = (
            network.gui_future_conditioner(future_x0.unsqueeze(0))
            if video_only
            else network.gui_plan_future_conditioner(plan_x0.unsqueeze(0), future_x0.unsqueeze(0))
        )
        adapted = network.gui_joint_ar_bridge(
            action_hidden,
            joint,
            schedule_scale=bridge_scale,
            detach_context=False,
        )
        logits = network.language_model.lm_head(adapted)
        base_logits = network.language_model.lm_head(action_hidden).detach()
    output_dict["gui_ar_logits"] = logits
    output_dict["gui_ar_base_logits"] = base_logits
    output_dict["gui_ar_max_logit_delta"] = (logits.detach() - base_logits).abs().max()
    with torch.no_grad():
        residual_norm = (adapted.float() - action_hidden.float()).norm(dim=-1).mean()
        base_norm = action_hidden.float().norm(dim=-1).mean().clamp_min(1e-8)
        output_dict["gui_ar_residual_norm_ratio"] = residual_norm / base_norm
    output_dict["gui_ar_labels"] = labels.unsqueeze(0)
    output_dict["gui_ar_token_mask"] = token_mask.unsqueeze(0)


@torch.no_grad()
def hybrid_ar_generate(network, request):
    """Greedy native tool-call decoding from sampled plan and future latents.

    Called through ``network.forward`` so FSDP materializes the original
    GUI-Libra reasoner and its adapter for the whole AR generation pass.
    """
    from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
        prepare_multimodal_reasoner_inputs,
    )

    device = next(network.language_model.parameters()).device
    sample = {
        key: value.to(device)
        for key, value in request["prompt"].items()
        if key in ("input_ids", "pixel_values", "image_grid_thw", "attention_mask")
    }
    if "prepared_prompt" in request:
        prompt = request["prepared_prompt"].to(device)
        visual_mask = request["prepared_visual_mask"].to(device)
        deepstack = [item.to(device) for item in request["prepared_deepstack"]]
        position_ids = request["prepared_position_ids"].to(device)
        mrope_deltas = request["prepared_mrope_deltas"].to(device)
    elif "precomputed_image_embeds" in request:
        from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import get_placeholder_mask, get_rope_index

        input_ids = sample["input_ids"]
        prompt = network.language_model.model.embed_tokens(input_ids).clone()
        image_embeds = torch.cat(request["precomputed_image_embeds"], dim=0).to(prompt)
        image_mask, _ = get_placeholder_mask(
            network.language_model, input_ids, inputs_embeds=prompt, image_features=image_embeds
        )
        prompt = prompt.masked_scatter(image_mask, image_embeds)
        visual_mask = image_mask[..., 0]
        deepstack = [item.to(prompt) for item in request["precomputed_deepstack"]]
        position_ids, mrope_deltas = get_rope_index(
            network.language_model,
            input_ids=input_ids,
            image_grid_thw=sample["image_grid_thw"],
            video_grid_thw=None,
            attention_mask=sample.get("attention_mask"),
        )
    else:
        # Match native GUI-Libra visual preprocessing.  The enclosing MoT
        # sampler may have BF16 autocast active; its vision tower/position
        # calculations must retain the native operation dtypes.
        with torch.autocast(device_type=device.type, enabled=False):
            prompt, visual_mask, deepstack, position_ids, mrope_deltas = prepare_multimodal_reasoner_inputs(
                network.language_model, **sample
            )
    compute_dtype = torch.bfloat16
    prompt = prompt.to(compute_dtype)
    deepstack = [item.to(compute_dtype) for item in deepstack]
    ar_only = bool(request.get("ar_only", False))
    video_only = bool(request.get("video_only", False))
    context = None
    if not ar_only:
        future = request["future"].to(device)
        if future.ndim == 5 and future.shape[0] == 1:
            future = future[0]
        if future.ndim != 4:
            raise ValueError("Sampled future must be [C,T,H,W]")
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            if video_only:
                context = network.gui_future_conditioner(future.unsqueeze(0))
            else:
                plan = request["plan"].to(device)
                if plan.ndim != 2:
                    raise ValueError("Sampled plan must be [T,D]")
                context = network.gui_plan_future_conditioner(plan.unsqueeze(0), future.unsqueeze(0))
    cache = ReasonerKVCache.empty(len(network.language_model.model.layers))
    generated = []
    stop_token_ids = {int(request["eos_token_id"])}
    stop_token_ids.update(int(value) for value in request.get("stop_token_ids", ()))
    stop_sequences = [
        tuple(int(value) for value in sequence) for sequence in request.get("stop_sequences", ()) if sequence
    ]
    import os

    if os.environ.get("GUI_AR_NATIVE_STOP_ONLY") == "1":
        # Native GUI-Libra generate stops at EOS / <|im_end|>, rather than
        # stopping immediately after the textual </answer> sequence.
        stop_sequences = []
    first_max_delta = None
    finished = False
    import os

    trace_file = os.environ.get("GUI_AR_TRACE_FILE")
    trace_logits = []
    trace_cache0_k = None
    trace_cache0_v = None
    trace_first_hidden = None
    import torch.distributed as dist

    distributed_lockstep = dist.is_initialized() and dist.get_world_size() > 1
    native_no_cache = os.environ.get("GUI_AR_NATIVE_NO_CACHE") == "1"
    for _ in range(int(request.get("max_new_tokens", 512))):
        if native_no_cache:
            if generated:
                text_model = network.language_model.model
                generated_ids = torch.tensor([generated], device=device, dtype=torch.long)
                generated_embeds = text_model.embed_tokens(generated_ids)
                full_prompt = torch.cat((prompt, generated_embeds), dim=1)
                generated_mask = torch.zeros((prompt.shape[0], len(generated)), device=device, dtype=torch.bool)
                full_visual_mask = torch.cat((visual_mask, generated_mask), dim=1)
                generated_positions = mrope_deltas.long().unsqueeze(0).expand(3, -1, -1) + torch.arange(
                    prompt.shape[1],
                    prompt.shape[1] + len(generated),
                    device=device,
                    dtype=torch.long,
                ).view(1, 1, -1)
                full_positions = torch.cat((position_ids, generated_positions), dim=-1)
            else:
                full_prompt = prompt
                full_visual_mask = visual_mask
                full_positions = position_ids
            forward_kwargs = {
                "input_ids": None,
                "cache": None,
                "inputs_embeds": full_prompt,
                "position_ids": full_positions,
                "visual_pos_masks": full_visual_mask,
                "deepstack_visual_embeds": deepstack,
            }
        elif generated:
            step_ids = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            step_positions = mrope_deltas.long().unsqueeze(0).expand(3, -1, -1) + cache.seq_len
            forward_kwargs = {
                "input_ids": step_ids,
                "cache": cache,
                "position_ids": step_positions,
            }
        else:
            forward_kwargs = {
                "input_ids": None,
                "cache": cache,
                "inputs_embeds": prompt,
                "position_ids": position_ids,
                "visual_pos_masks": visual_mask,
                "deepstack_visual_embeds": deepstack,
            }
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            hidden = network.language_model.model.reasoner_forward(**forward_kwargs)[:, -1:]
            if trace_file and trace_first_hidden is None:
                trace_first_hidden = hidden.detach().float().cpu()
            base_logits = network.language_model.lm_head(hidden)
            logits = (
                base_logits
                if ar_only
                else network.language_model.lm_head(network.gui_joint_ar_bridge(hidden, context, schedule_scale=1.0))
            )
        if trace_file and len(trace_logits) < 64:
            trace_logits.append(base_logits[0, -1].detach().float().cpu())
            if trace_cache0_k is None and not native_no_cache:
                trace_cache0_k = cache.keys[0].detach().cpu()
                trace_cache0_v = cache.values[0].detach().cpu()
        if first_max_delta is None:
            first_max_delta = float((logits.float() - base_logits.float()).abs().max())
            if request.get("audit_only"):
                result = {
                    "first_base_logits": base_logits[0, -1].float().cpu(),
                    "first_joint_logits": logits[0, -1].float().cpu(),
                    "first_max_logit_delta": first_max_delta,
                }
                if request.get("audit_layers"):
                    text_model = network.language_model.model
                    audit_h = prompt
                    layer_last = [audit_h[0, -1].float().cpu()]
                    audit_cos, audit_sin = text_model.rotary_emb(audit_h, position_ids=position_ids)
                    for audit_idx, audit_layer in enumerate(text_model.layers):
                        audit_h = audit_layer.reasoner_forward(audit_h, audit_cos, audit_sin, None, audit_idx)
                        if audit_idx < len(deepstack):
                            audit_h = audit_h.clone()
                            audit_h[visual_mask, :] = audit_h[visual_mask, :] + deepstack[audit_idx]
                        layer_last.append(audit_h[0, -1].float().cpu())
                    audit_h = text_model.norm(audit_h)
                    layer_last.append(audit_h[0, -1].float().cpu())
                    result["layer_last"] = torch.stack(layer_last)
                    result["prompt_embeds"] = prompt.float().cpu()
                    result["deepstack"] = [item.float().cpu() for item in deepstack]
                    result["visual_mask"] = visual_mask.cpu()
                    result["position_ids"] = position_ids.cpu()
                return result
        if not finished:
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            if len(generated) % 32 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"hybrid_ar_generate tokens={len(generated)}", flush=True)
            finished = next_id in stop_token_ids or any(
                len(generated) >= len(sequence) and tuple(generated[-len(sequence) :]) == sequence
                for sequence in stop_sequences
            )
        if distributed_lockstep:
            any_active = torch.tensor([0 if finished else 1], device=device, dtype=torch.int32)
            dist.all_reduce(any_active, op=dist.ReduceOp.MAX)
            if not bool(any_active.item()):
                break
        elif finished:
            break
    if trace_file:
        torch.save(
            {
                "logits": torch.stack(trace_logits),
                "ids": generated,
                "cache0_k": trace_cache0_k,
                "cache0_v": trace_cache0_v,
                "first_hidden": trace_first_hidden,
                "inv_freq": network.language_model.model.rotary_emb.inv_freq.detach().cpu(),
                "position_ids": position_ids.detach().cpu(),
            },
            trace_file,
        )
    return {
        "ids": generated,
        "first_max_logit_delta": first_max_delta,
        "finished": finished,
    }
