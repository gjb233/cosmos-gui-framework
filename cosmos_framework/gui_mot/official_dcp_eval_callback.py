"""Official AndroidControl evaluation for a distributed MoT DCP checkpoint."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from cosmos_framework.utils.callback import Callback

from .eval_gui_libra_official_androidcontrol import (
    SYSTEM_PROMPT,
    extract_plan_fields,
    official_query,
)
from .joint_dataset import read_screen
from .joint_policy import JointPolicy, parse_native_tool_calls


def _official_fields(raw_text):
    fields = extract_plan_fields(raw_text)
    if fields["action_type"]:
        return fields
    parsed = parse_native_tool_calls(raw_text)
    if not parsed:
        return fields
    action = parsed[0]["action"]
    kind = action["type"]
    mapping = {
        "click": "Click",
        "long_press": "LongPress",
        "input_text": "Write",
        "scroll": "Scroll",
        "wait": "Wait",
        "navigate_back": "NavigateBack",
        "navigate_home": "Home",
        "open_app": "OpenApp",
    }
    point = None
    if kind in {"click", "long_press"}:
        point = [round(action["x"] * 1000), round(action["y"] * 1000)]
    value = "None"
    if kind == "input_text":
        value = action["text"]
    elif kind == "scroll":
        value = action["direction"]
    elif kind == "open_app":
        value = action["app_name"]
    return {
        "action_type": mapping.get(kind, kind),
        "element_description": "",
        "value": value,
        "point_2d": point,
    }


class OfficialAndroidControlDCPEval(Callback):
    def __init__(
        self,
        official_samples,
        screenshot_dir,
        output,
        target_height=640,
        target_width=384,
        max_new_tokens=512,
        sampling_steps=20,
        max_samples=0,
    ):
        super().__init__()
        self.rows = json.loads(Path(official_samples).read_text(encoding="utf-8"))
        if len(self.rows) != 398:
            raise ValueError(f"Expected 398 official samples, got {len(self.rows)}")
        independent_shards = int(os.environ.get("GUI_OFFICIAL_INDEPENDENT_NUM_SHARDS", "1"))
        independent_index = int(os.environ.get("GUI_OFFICIAL_INDEPENDENT_SHARD_INDEX", "0"))
        if independent_shards < 1 or not 0 <= independent_index < independent_shards:
            raise ValueError("invalid independent shard index/count")
        self.row_indices = list(range(len(self.rows)))[independent_index::independent_shards]
        max_samples = int(max_samples)
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        if max_samples:
            self.row_indices = self.row_indices[:max_samples]
        self.screenshot_dir = Path(screenshot_dir)
        self.output = Path(output)
        self.target_hw = (int(target_height), int(target_width))
        self.max_new_tokens = int(max_new_tokens)
        self.sampling_steps = int(sampling_steps)
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")

    @torch.no_grad()
    def on_train_start(self, model, iteration=0):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        if os.environ.get("GUI_REPLACE_VISUAL_WITH_HF", "0") == "1":
            from safetensors import safe_open
            from transformers import AutoConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

            backbone = Path(os.environ["GUI_BACKBONE_PATH"])
            top_config = AutoConfig.from_pretrained(backbone, local_files_only=True)
            top_config.vision_config._attn_implementation = "sdpa"
            replacement = Qwen3VLVisionModel._from_config(
                top_config.vision_config, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            weight_map = json.loads((backbone / "model.safetensors.index.json").read_text())["weight_map"]
            state = {}
            for source_name, shard in weight_map.items():
                if not source_name.startswith("model.visual."):
                    continue
                with safe_open(backbone / shard, framework="pt", device="cpu") as handle:
                    state[source_name.removeprefix("model.visual.")] = handle.get_tensor(source_name)
            replacement.load_state_dict(state, strict=True)
            device = next(model.net.language_model.visual.parameters()).device
            replacement = replacement.to(device=device, dtype=torch.bfloat16).eval()
            model.net.language_model.visual = replacement
            print(json.dumps({"hf_visual_replacement_parameters": len(state), "rank": rank}), flush=True)
        if os.environ.get("GUI_RELOAD_VISUAL_FROM_BACKBONE", "0") == "1":
            from safetensors import safe_open

            backbone = Path(os.environ["GUI_BACKBONE_PATH"])
            weight_map = json.loads((backbone / "model.safetensors.index.json").read_text())["weight_map"]
            handles = {}
            loaded = 0
            try:
                for name, parameter in model.net.language_model.visual.named_parameters():
                    source_name = "model.visual." + name
                    shard = weight_map[source_name]
                    if shard not in handles:
                        handles[shard] = safe_open(backbone / shard, framework="pt", device="cpu")
                    source = handles[shard].get_tensor(source_name)
                    parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
                    loaded += 1
            finally:
                handles.clear()
            print(json.dumps({"visual_parameters_reloaded": loaded, "rank": rank}), flush=True)
        steps = math.ceil(len(self.row_indices) / world)
        shard_dir = self.output / f"shard-{rank}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        output_path = shard_dir / "predictions.jsonl"
        if output_path.exists():
            raise FileExistsError(output_path)
        training = model.training
        model.eval()
        completed = 0
        parse_errors = 0
        with output_path.open("x", encoding="utf-8", buffering=1) as stream:
            for local_index in range(steps):
                shard_index = local_index * world + rank
                real = shard_index < len(self.row_indices)
                index = self.row_indices[shard_index] if real else self.row_indices[0]
                row = self.rows[index]
                image_path = self.screenshot_dir / row["screenshot"]
                with Image.open(image_path) as source:
                    original_size = source.size
                    prefix_image = (
                        torch.from_numpy(np.array(source.convert("RGB"), copy=True)).permute(2, 0, 1).contiguous()
                    )
                current = read_screen(image_path, self.target_hw)
                sample_query = official_query(row, original_size)
                prediction = JointPolicy(model).predict_action(
                    current,
                    sample_query,
                    system_prompt=SYSTEM_PROMPT,
                    exact_user_prompt=True,
                    prefix_image=prefix_image,
                    seed=index,
                    sampling_steps=self.sampling_steps,
                    horizon=1,
                    return_action_values=True,
                    return_future_latents=True,
                    max_new_tokens=self.max_new_tokens,
                )
                audit_reference = os.environ.get("GUI_OFFICIAL_AUDIT_REFERENCE")
                if audit_reference and completed == 0:
                    tokenizer = model._gui_processor().tokenizer
                    op_capture = {}
                    op_hooks = []
                    layer0 = model.net.language_model.model.layers[0]
                    for op_name, op_module in (
                        ("input_norm", layer0.input_layernorm),
                        ("q_proj", layer0.self_attn.q_proj),
                        ("k_proj", layer0.self_attn.k_proj),
                        ("v_proj", layer0.self_attn.v_proj),
                        ("q_norm", layer0.self_attn.q_norm),
                        ("k_norm", layer0.self_attn.k_norm),
                        ("o_proj", layer0.self_attn.o_proj),
                        ("post_norm", layer0.post_attention_layernorm),
                        ("mlp", layer0.mlp),
                    ):

                        def save_op(_module, _inputs, output, name=op_name):
                            if name not in op_capture:
                                if isinstance(output, tuple):
                                    output = output[0]
                                op_capture[name] = output.detach().float().cpu()

                        op_hooks.append(op_module.register_forward_hook(save_op))
                    audit_request = {
                        "prompt": model._gui_prefix[0],
                        "ar_only": True,
                        "eos_token_id": tokenizer.eos_token_id,
                        "audit_only": True,
                        "audit_layers": True,
                    }
                    native_prefix_file = os.environ.get("GUI_NATIVE_PREFIX_FILE")
                    if native_prefix_file:
                        native_prefix = torch.load(native_prefix_file, map_location="cpu", weights_only=False)
                        audit_request.update(
                            prepared_prompt=native_prefix["native_hidden0"],
                            prepared_visual_mask=native_prefix["native_visual_mask"],
                            prepared_deepstack=native_prefix["native_deepstack"],
                            prepared_position_ids=native_prefix["native_position_ids"],
                            prepared_mrope_deltas=native_prefix["native_mrope_deltas"],
                        )
                    audit = model.net(packed_seq=SimpleNamespace(gui_ar_decode_request=audit_request))
                    for op_hook in op_hooks:
                        op_hook.remove()
                    reference = torch.load(audit_reference, map_location="cpu", weights_only=True)
                    prompt_ids = model._gui_prefix[0]["input_ids"].cpu()
                    base_logits = audit["first_base_logits"].float()
                    original_logits = reference["logits"].float()
                    top_values, top_ids = torch.topk(original_logits, 5)
                    report = {
                        "prompt_ids_equal": torch.equal(reference["prompt_ids"], prompt_ids),
                        "prompt_shape_reference": list(reference["prompt_ids"].shape),
                        "prompt_shape_mot": list(prompt_ids.shape),
                        "max_abs_logit_diff": float((original_logits - base_logits).abs().max()),
                        "mean_abs_logit_diff": float((original_logits - base_logits).abs().mean()),
                        "original_top_id": int(original_logits.argmax()),
                        "mot_top_id": int(base_logits.argmax()),
                        "original_top5_ids": top_ids.tolist(),
                        "original_top5_logits": top_values.tolist(),
                        "mot_logits_at_original_top5": base_logits[top_ids].tolist(),
                    }
                    if "layer_last" in reference and "layer_last" in audit:
                        ref_layers = reference["layer_last"].float()
                        mot_layers = audit["layer_last"].float()
                        report["layer_shapes"] = [list(ref_layers.shape), list(mot_layers.shape)]
                        pairs = None
                        if ref_layers.shape == mot_layers.shape:
                            pairs = list(zip(ref_layers, mot_layers))
                        elif (
                            ref_layers.shape[0] + 1 == mot_layers.shape[0]
                            and ref_layers.shape[1:] == mot_layers.shape[1:]
                        ):
                            pairs = list(zip(ref_layers[:-1], mot_layers[: ref_layers.shape[0] - 1]))
                            pairs.append((ref_layers[-1], mot_layers[-1]))
                        if pairs is not None:
                            report["layers"] = [
                                {
                                    "index": i,
                                    "max_abs": float((r - m).abs().max()),
                                    "mean_abs": float((r - m).abs().mean()),
                                    "cosine": float(torch.nn.functional.cosine_similarity(r, m, dim=0)),
                                }
                                for i, (r, m) in enumerate(pairs)
                            ]
                    if "hidden0" in reference and "prompt_embeds" in audit:
                        ref_h0 = reference["hidden0"].float()
                        mot_h0 = audit["prompt_embeds"].float()
                        report["hidden0_shapes"] = [list(ref_h0.shape), list(mot_h0.shape)]
                        if ref_h0.shape == mot_h0.shape:
                            h0_diff = (ref_h0 - mot_h0).abs()
                            mask = audit["visual_mask"].bool()
                            report["hidden0_max_abs"] = float(h0_diff.max())
                            report["hidden0_mean_abs"] = float(h0_diff.mean())
                            report["hidden0_visual_max_abs"] = float(h0_diff[mask].max())
                            report["hidden0_visual_mean_abs"] = float(h0_diff[mask].mean())
                            report["hidden0_text_max_abs"] = float(h0_diff[~mask].max())
                            report["hidden0_text_mean_abs"] = float(h0_diff[~mask].mean())
                    ref_h0 = reference.get("hidden0")
                    ref_layer0 = reference.get("layer0")
                    if ref_h0 is not None and ref_layer0 is not None and "position_ids" in audit:
                        text_model = model.net.language_model.model
                        ref_h0_cuda = ref_h0.to(device=next(text_model.parameters()).device, dtype=torch.bfloat16)
                        ref_cos, ref_sin = text_model.rotary_emb(
                            ref_h0_cuda, position_ids=audit["position_ids"].to(ref_h0_cuda.device)
                        )
                        mot_from_ref = (
                            text_model.layers[0].reasoner_forward(ref_h0_cuda, ref_cos, ref_sin, None, 0).float().cpu()
                        )
                        ref_layer0 = ref_layer0.float()
                        delta = (ref_layer0 - mot_from_ref).abs()
                        report["layer0_from_reference_hidden"] = {
                            "max_abs": float(delta.max()),
                            "mean_abs": float(delta.mean()),
                            "last_token_max_abs": float(delta[:, -1].max()),
                            "last_token_mean_abs": float(delta[:, -1].mean()),
                        }
                    report["op_diffs"] = {}
                    for op_name, mot_op in op_capture.items():
                        if op_name not in reference:
                            continue
                        ref_op = reference[op_name].float()
                        item = {"reference_shape": list(ref_op.shape), "mot_shape": list(mot_op.shape)}
                        if ref_op.shape == mot_op.shape:
                            delta = (ref_op - mot_op).abs()
                            item.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
                            if delta.ndim >= 3:
                                item.update(
                                    last_token_max_abs=float(delta[:, -1].max()),
                                    last_token_mean_abs=float(delta[:, -1].mean()),
                                )
                        report["op_diffs"][op_name] = item
                    for key in ("pixel_values", "image_grid_thw"):
                        if key in reference and key in model._gui_prefix[0]:
                            r = reference[key].cpu()
                            m = (
                                model._gui_prefix[0][key].float().cpu()
                                if r.is_floating_point()
                                else model._gui_prefix[0][key].cpu()
                            )
                            report[key + "_shape_reference"] = list(r.shape)
                            report[key + "_shape_mot"] = list(m.shape)
                            report[key + "_equal"] = bool(torch.equal(r, m))
                            if r.is_floating_point() and r.shape == m.shape:
                                report[key + "_max_abs"] = float((r.float() - m.float()).abs().max())
                    lora_b = [(n, p.detach().float()) for n, p in model.net.named_parameters() if "lora_B" in n]
                    report["lora_b_count"] = len(lora_b)
                    report["lora_b_max_abs"] = max((float(v.abs().max()) for _, v in lora_b), default=0.0)
                    from safetensors import safe_open

                    backbone = Path(os.environ["GUI_BACKBONE_PATH"])
                    weight_map = json.loads((backbone / "model.safetensors.index.json").read_text())["weight_map"]
                    report["vision_attn_implementation"] = str(
                        model.net.language_model.visual.config._attn_implementation
                    )
                    from transformers import AutoConfig

                    native_vision_cfg = AutoConfig.from_pretrained(
                        backbone, local_files_only=True
                    ).vision_config.to_dict()
                    runtime_vision_cfg = model.net.language_model.visual.config.to_dict()
                    report["vision_config_diffs"] = {
                        key: {"native": native_vision_cfg.get(key), "runtime": runtime_vision_cfg.get(key)}
                        for key in sorted(set(native_vision_cfg) | set(runtime_vision_cfg))
                        if native_vision_cfg.get(key) != runtime_vision_cfg.get(key)
                    }
                    report["visual_buffers"] = {
                        name: {
                            "shape": list(buffer.shape),
                            "dtype": str(buffer.dtype),
                            "min": float(buffer.float().min()),
                            "max": float(buffer.float().max()),
                        }
                        for name, buffer in model.net.language_model.visual.named_buffers()
                        if buffer.numel()
                    }
                    for report_name, source_name, runtime_tensor in (
                        ("final_norm", "model.language_model.norm.weight", model.net.language_model.model.norm.weight),
                        ("lm_head", "lm_head.weight", model.net.language_model.lm_head.weight),
                        (
                            "vision_patch",
                            "model.visual.patch_embed.proj.weight",
                            model.net.language_model.visual.patch_embed.proj.weight,
                        ),
                        (
                            "vision_block0_qkv",
                            "model.visual.blocks.0.attn.qkv.weight",
                            model.net.language_model.visual.blocks[0].attn.qkv.weight,
                        ),
                        (
                            "vision_block0_norm1",
                            "model.visual.blocks.0.norm1.weight",
                            model.net.language_model.visual.blocks[0].norm1.weight,
                        ),
                        (
                            "vision_merger_fc2",
                            "model.visual.merger.linear_fc2.weight",
                            model.net.language_model.visual.merger.linear_fc2.weight,
                        ),
                    ):
                        with safe_open(backbone / weight_map[source_name], framework="pt", device="cpu") as handle:
                            source_tensor = handle.get_tensor(source_name).float()
                        runtime_cpu = runtime_tensor.detach().float().cpu()
                        report[report_name + "_max_abs_weight_diff"] = float((source_tensor - runtime_cpu).abs().max())
                        report[report_name + "_source_abs_max"] = float(source_tensor.abs().max())
                        report[report_name + "_runtime_abs_max"] = float(runtime_cpu.abs().max())
                    if os.environ.get("GUI_AUDIT_VISUAL_SIDECAR") == "1":
                        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

                        native_cfg = AutoConfig.from_pretrained(backbone, local_files_only=True).vision_config
                        native_cfg._attn_implementation = "sdpa"
                        sidecar = Qwen3VLVisionModel._from_config(
                            native_cfg, dtype=torch.bfloat16, attn_implementation="sdpa"
                        )
                        source_state = {}
                        for source_name, shard in weight_map.items():
                            if source_name.startswith("model.visual."):
                                with safe_open(backbone / shard, framework="pt", device="cpu") as handle:
                                    source_state[source_name.removeprefix("model.visual.")] = handle.get_tensor(
                                        source_name
                                    )
                        sidecar.load_state_dict(source_state, strict=True)
                        vision_device = next(model.net.language_model.visual.parameters()).device
                        sidecar = sidecar.to(device=vision_device, dtype=torch.bfloat16).eval()
                        pixels = model._gui_prefix[0]["pixel_values"].to(vision_device)
                        grid = model._gui_prefix[0]["image_grid_thw"].to(vision_device)
                        captures = {"native": {}, "runtime": {}}
                        capture_hooks = []
                        for label, vision in (("native", sidecar), ("runtime", model.net.language_model.visual)):
                            for stage_name, stage_module in (
                                ("patch", vision.patch_embed),
                                ("rotary", vision.rotary_pos_emb),
                                ("block0", vision.blocks[0]),
                                ("block1", vision.blocks[1]),
                                ("block8", vision.blocks[8]),
                                ("block16", vision.blocks[16]),
                                ("block_last", vision.blocks[-1]),
                                ("merger", vision.merger),
                            ):

                                def capture(_module, _args, value, group=label, name=stage_name):
                                    if isinstance(value, tuple):
                                        value = value[0]
                                    captures[group][name] = value.detach().float().cpu()

                                capture_hooks.append(stage_module.register_forward_hook(capture))
                        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                            native_visual, native_deep = sidecar(pixels, grid)
                            runtime_visual, runtime_deep = model.net.language_model.visual(pixels, grid)
                        for capture_hook in capture_hooks:
                            capture_hook.remove()
                        report["visual_stage_diffs"] = {}
                        for stage_name, native_stage in captures["native"].items():
                            runtime_stage = captures["runtime"][stage_name]
                            report["visual_stage_diffs"][stage_name] = {
                                "shapes": [list(native_stage.shape), list(runtime_stage.shape)],
                                "max": float((native_stage - runtime_stage).abs().max()),
                                "mean": float((native_stage - runtime_stage).abs().mean()),
                            }
                        full_native_stages = torch.load(
                            "/rscratch/wzzheng/code/cosmos-gui/outputs/libra-overfit-data/native-official-index0-visual-stages.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        report["full_native_stage_diffs"] = {
                            name: float((full_native_stages[name] - captures["runtime"][name]).abs().max())
                            for name in full_native_stages
                        }
                        native_buffer = dict(sidecar.named_buffers())["rotary_pos_emb.inv_freq"].float().cpu()
                        runtime_buffer = (
                            dict(model.net.language_model.visual.named_buffers())["rotary_pos_emb.inv_freq"]
                            .float()
                            .cpu()
                        )
                        report["visual_rotary_buffer_max_abs"] = float((native_buffer - runtime_buffer).abs().max())
                        runtime_rotary = model.net.language_model.visual.rotary_pos_emb
                        runtime_formula = 1.0 / (
                            runtime_rotary.theta
                            ** (torch.arange(0, runtime_rotary.dim, 2, dtype=torch.float32) / runtime_rotary.dim)
                        )
                        report["visual_rotary_details"] = {
                            "native_class": str(type(sidecar.rotary_pos_emb)),
                            "runtime_class": str(type(runtime_rotary)),
                            "runtime_theta": runtime_rotary.theta,
                            "runtime_dim": runtime_rotary.dim,
                            "native_buffer_first": native_buffer[:8].tolist(),
                            "runtime_buffer_first": runtime_buffer[:8].tolist(),
                            "formula_first": runtime_formula[:8].tolist(),
                            "formula_vs_native_max": float((runtime_formula - native_buffer).abs().max()),
                            "formula_vs_runtime_max": float((runtime_formula - runtime_buffer).abs().max()),
                            "native_output_first": captures["native"]["rotary"][1, :8].tolist(),
                            "runtime_output_first": captures["runtime"]["rotary"][1, :8].tolist(),
                        }
                        report["sidecar_output_shapes"] = [list(native_visual.shape), list(runtime_visual.shape)]
                        visual_delta = (native_visual.float() - runtime_visual.float()).abs()
                        report["sidecar_visual_max_abs"] = float(visual_delta.max())
                        report["sidecar_visual_mean_abs"] = float(visual_delta.mean())
                        if "native_hidden0" in reference and "native_visual_mask" in reference:
                            saved_visual = (
                                reference["native_hidden0"][reference["native_visual_mask"].bool()].float().cpu()
                            )
                            report["saved_native_visual_shape"] = list(saved_visual.shape)
                            if saved_visual.shape == native_visual.shape:
                                report["saved_native_vs_sidecar_max_abs"] = float(
                                    (saved_visual - native_visual.float().cpu()).abs().max()
                                )
                                report["saved_native_vs_runtime_max_abs"] = float(
                                    (saved_visual - runtime_visual.float().cpu()).abs().max()
                                )
                        report["sidecar_deep_max_abs"] = [
                            float((a.float() - b.float()).abs().max()) for a, b in zip(native_deep, runtime_deep)
                        ]
                        report["sidecar_weight_max_abs"] = max(
                            float((source_state[name].float().cpu() - param.detach().float().cpu()).abs().max())
                            for name, param in model.net.language_model.visual.named_parameters()
                        )
                        del sidecar, source_state
                        torch.cuda.empty_cache()
                    (shard_dir / "first_logit_audit.json").write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                if not real:
                    continue
                raw_text = prediction["raw_text"]
                fields = _official_fields(raw_text)
                parse_errors += int(not fields["action_type"])
                result = {
                    "_index": index,
                    "episode_id": row["episode_id"],
                    "step": row["step"],
                    "instruction": row["step_instruction"],
                    "action": "",
                    **fields,
                    "reason": "",
                    "response": raw_text,
                    "finish_reason": "stop" if prediction["complete"] else "length",
                    "output_tokens": len(model._gui_processor().tokenizer.encode(raw_text)),
                    "fm_sampling_steps": self.sampling_steps,
                    "first_max_logit_delta": prediction["first_max_logit_delta"],
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                (shard_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "expected": len(range(rank, len(self.row_indices), world)),
                            "evaluated": completed,
                            "parse_errors": parse_errors,
                            "iteration": iteration,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        model.train(training)
        if dist.is_initialized():
            dist.barrier()
        raise SystemExit(0)
