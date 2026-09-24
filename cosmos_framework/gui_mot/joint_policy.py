"""Action-only output from native JOINT action/vision denoising."""

import json
import re
from dataclasses import asdict
from types import SimpleNamespace

import torch

from .action_codec import decode_action
from .joint_dataset import make_sample, read_screen


class JointPolicy:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def predict_action(
        self,
        current_image,
        instruction,
        *,
        seed=42,
        sampling_steps=20,
        decode_future=False,
        return_future_latents=False,
        return_action_values=False,
        horizon=1,
        max_new_tokens=512,
        system_prompt=None,
        exact_user_prompt=False,
        prefix_image=None,
    ):
        from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn

        if type(seed) is not int or type(sampling_steps) is not int or sampling_steps < 1:
            raise ValueError("Expected integer seed and positive sampling_steps")
        current = read_screen(current_image) if not isinstance(current_image, torch.Tensor) else current_image
        sample = make_sample(
            current,
            instruction,
            max_action_dim=self.model.config.max_action_dim,
            horizon=horizon,
        )
        if exact_user_prompt:
            sample["ai_caption"] = instruction
        if system_prompt:
            sample["gui_system_prompt"] = system_prompt
        if prefix_image is not None:
            if (
                not isinstance(prefix_image, torch.Tensor)
                or prefix_image.ndim != 3
                or prefix_image.shape[0] != 3
                or prefix_image.dtype != torch.uint8
            ):
                raise ValueError("prefix_image must be a uint8 CHW tensor")
            sample["gui_current_screens"] = [prefix_image.detach().cpu().clone()]
        batch = custom_collate_fn([sample])
        if getattr(self.model, "hybrid_ar", False) or getattr(self.model, "mot_joint", False):
            from cosmos_framework.data.generator.action.utils.action_processing import (
                ActionProcessingRecord,
            )

            # The native sampler returns externalized actions. For a semantic
            # plan its model-space width is the output, so retain all 64 slots.
            batch["action_processing_record"] = [
                ActionProcessingRecord(
                    raw_action_dim=self.model.config.max_action_dim,
                    action_normalizer=None,
                )
            ]
            batch["raw_action_dim"] = [torch.tensor(self.model.config.max_action_dim)]
        # No GT action, future image, or applicability mask enters this call.
        result = self.model.generate_samples_from_batch(
            batch,
            seed=[seed],
            num_steps=sampling_steps,
            guidance=1.0,
            use_batched_cfg=False,
            upsample_task=None,
        )
        if len(result.get("action", [])) != 1 or len(result.get("vision", [])) != 1:
            raise RuntimeError("Joint sampler must return one action and one latent video")
        action_tensor = result["action"][0]
        if action_tensor.ndim != 2 or action_tensor.shape[0] != horizon:
            raise RuntimeError(f"Expected {horizon} GUI action tokens")
        if getattr(self.model, "hybrid_ar", False) or getattr(self.model, "mot_joint", False):
            tokenizer = self.model._gui_processor().tokenizer
            request = {
                "prompt": self.model._gui_prefix[0],
                # Training reconstructs x0 from the denoiser and conditions
                # AR logits through this same bridge. Inference must therefore
                # consume the final FM plan/future samples as well.
                "ar_only": False,
                "eos_token_id": tokenizer.eos_token_id,
                "stop_token_ids": [tokenizer.convert_tokens_to_ids("<|im_end|>")],
                "stop_sequences": [
                    tokenizer.encode("</answer>", add_special_tokens=False),
                    tokenizer.encode("</tool_call>", add_special_tokens=False),
                ],
                "max_new_tokens": int(max_new_tokens),
            }
            request["plan"] = action_tensor
            request["future"] = result["vision"][0]
            native_prefix_file = __import__("os").environ.get("GUI_NATIVE_PREFIX_FILE")
            if native_prefix_file:
                native_prefix = torch.load(native_prefix_file, map_location="cpu", weights_only=False)
                request["prepared_prompt"] = native_prefix["native_hidden0"]
                request["prepared_visual_mask"] = native_prefix["native_visual_mask"]
                request["prepared_deepstack"] = native_prefix["native_deepstack"]
                request["prepared_position_ids"] = native_prefix["native_position_ids"]
                request["prepared_mrope_deltas"] = native_prefix["native_mrope_deltas"]
            if __import__("os").environ.get("GUI_PRECOMPUTE_VISUAL_OUTSIDE_FSDP", "0") == "1":
                from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import get_image_features

                language_model = self.model.net.language_model
                device = next(language_model.visual.parameters()).device
                prefix = request["prompt"]
                pixels = prefix["pixel_values"].to(device)
                grid = prefix["image_grid_thw"].to(device)
                image_embeds, deepstack = get_image_features(language_model, pixels, grid)
                request["precomputed_image_embeds"] = [item.detach() for item in image_embeds]
                request["precomputed_deepstack"] = [item.detach() for item in deepstack]
            generated = self.model.net(packed_seq=SimpleNamespace(gui_ar_decode_request=request))
            raw_text = self.model._gui_processor().tokenizer.decode(generated["ids"], skip_special_tokens=False)
            parsed = parse_native_tool_calls(raw_text)
            output = {
                "actions": parsed[:horizon],
                "raw_text": raw_text,
                "first_max_logit_delta": generated["first_max_logit_delta"],
                "complete": bool(generated.get("finished", False)),
            }
            if return_action_values:
                output["plan_values"] = action_tensor.detach().float().cpu().tolist()
            if return_future_latents:
                output["future_latents"] = result["vision"][0]
            if decode_future:
                latent = result["vision"][0]
                output["future_video"] = self.model.decode(latent.unsqueeze(0) if latent.ndim == 4 else latent)
            return output
        decoded = [asdict(decode_action(item)) for item in action_tensor]
        output = decoded[0] if horizon == 1 else {"actions": decoded}
        if return_action_values:
            values = action_tensor.detach().float().cpu().tolist()
            output["action_values"] = values[0] if horizon == 1 else values
        if return_future_latents:
            output["future_latents"] = result["vision"][0]
        if decode_future:
            latent = result["vision"][0]
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            output["future_video"] = self.model.decode(latent)
        return output


def parse_native_tool_calls(text):
    """Parse GUI-Libra action payloads without fabricating missing actions."""
    actions = []
    for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            payload = json.loads(body)
            if payload.get("name") != "mobile_use":
                continue
            args = payload["arguments"]
            kind = args["action"]
            if kind in ("click", "long_press"):
                x, y = args["coordinate"]
                action = {"type": kind, "x": float(x) / 999, "y": float(y) / 999}
            elif kind == "swipe":
                inverse = {"down": "up", "up": "down", "right": "left", "left": "right"}
                action = {"type": "scroll", "direction": inverse[args["direction"]]}
            elif kind == "type":
                action = {"type": "input_text", "text": args["text"]}
            elif kind == "open":
                action = {"type": "open_app", "app_name": args["text"]}
            elif kind == "system_button":
                action = {"type": "navigate_" + args["button"]}
            elif kind == "wait":
                action = {"type": "wait"}
            else:
                continue
            actions.append({"action": action, "valid": True, "payload_complete": True})
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if actions:
        return actions
    # GUI-Libra may emit standalone JSON action objects or an actions array.
    # Score only completed objects when generation stops mid-trajectory.
    for body in re.findall(r'\{[^{}]*"action_type"[^{}]*\}', text, re.S):
        try:
            item = json.loads(body)
            kind = item["action_type"]
            if kind in ("click", "long_press"):
                x, y = item.get("target_coordinate", item.get("coordinate"))
                action = {"type": kind, "x": float(x) / 999, "y": float(y) / 999}
            elif kind in ("scroll", "swipe"):
                direction = item.get("direction")
                if direction not in ("up", "down", "left", "right"):
                    description = item.get("action_description", "").lower()
                    direction = next(
                        (word for word in ("up", "down", "left", "right") if re.search(rf"\b{word}\b", description)),
                        None,
                    )
                if direction is None:
                    continue
                action = {"type": "scroll", "direction": direction}
            elif kind in ("back", "home"):
                action = {"type": "navigate_" + kind}
            elif kind in ("navigate_back", "navigate_home", "wait"):
                action = {"type": kind}
            elif kind in ("input_text", "type"):
                action = {"type": "input_text", "text": item.get("target_text") or item["text"]}
            elif kind == "open_app":
                action = {"type": "open_app", "app_name": item["target_app_name"]}
            else:
                continue
            actions.append({"action": action, "valid": True, "payload_complete": True})
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return actions
