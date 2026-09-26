"""Native GUI-Libra action text and protected hybrid AR components."""

import json
from dataclasses import dataclass

import torch
from torch import nn


def action_tool_call(action):
    """Serialize a canonical action with GUI-Libra's original mobile_use schema."""
    kind = action["type"]
    args = {}
    if kind in {"click", "long_press"}:
        args = {
            "action": kind,
            "coordinate": [round(action["x"] * 999), round(action["y"] * 999)],
        }
    elif kind == "scroll":
        args = {
            "action": "swipe",
            "direction": {"up": "down", "down": "up", "left": "right", "right": "left"}[action["direction"]],
        }
    elif kind == "input_text":
        args = {"action": "type", "text": action["text"]}
    elif kind == "open_app":
        args = {"action": "open", "text": action["app_name"]}
    elif kind in {"navigate_home", "navigate_back"}:
        args = {
            "action": "system_button",
            "button": kind.removeprefix("navigate_"),
        }
    elif kind == "wait":
        args = {"action": "wait"}
    else:
        raise ValueError(f"Unsupported action: {kind}")
    payload = {"name": "mobile_use", "arguments": args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def action_trajectory_text(actions):
    if not actions:
        raise ValueError("Action trajectory cannot be empty")
    return "\n".join(action_tool_call(action) for action in actions)


def libra_json_action(action):
    """One GUI-Libra native JSON action object."""
    kind = action["type"]
    item = {"action_description": kind.replace("_", " "), "action_type": kind}
    if kind in {"click", "long_press"}:
        item["target_coordinate"] = [round(action["x"] * 999), round(action["y"] * 999)]
    elif kind == "scroll":
        item["direction"] = action["direction"]
        item["action_description"] = f"Scroll {action['direction']} on the current screen."
    elif kind == "input_text":
        item["text"] = action["text"]
    elif kind == "open_app":
        item["target_app_name"] = action["app_name"]
    elif kind not in {"navigate_home", "navigate_back", "wait"}:
        raise ValueError(f"Unsupported action: {kind}")
    return json.dumps(item, ensure_ascii=False, indent=4)


def libra_json_trajectory(actions):
    """Match GUI-Libra's sequence of standalone JSON action objects."""
    if not actions:
        raise ValueError("Action trajectory cannot be empty")
    return "\n".join(libra_json_action(action) for action in actions)


@dataclass(frozen=True)
class HybridStage:
    name: str
    bridge_scale: float
    detach_joint: bool
    enable_ar_loss: bool
    enable_gui_lora: bool


def hybrid_stage(step, *, align_steps=500, bridge_steps=1000):
    """One-run schedule: align joint latents, train bridge, then joint LoRA."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if align_steps < 1 or bridge_steps < 1:
        raise ValueError("stage lengths must be positive")
    if step < align_steps:
        return HybridStage("align", 0.0, True, False, False)
    if step < align_steps + bridge_steps:
        progress = (step - align_steps + 1) / bridge_steps
        return HybridStage("bridge", min(progress, 1.0), True, True, False)
    return HybridStage("joint", 1.0, False, True, True)


class ZeroInitJointPrefix(nn.Module):
    """Residual soft-prefix adapter that is exactly inert at initialization."""

    def __init__(self, hidden_size, bottleneck=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        # The zero-initialized up projection makes the residual exactly inert.
        # Keep the multiplier nonzero so the projection receives a gradient on
        # the first bridge-training step (zeroing both factors is a dead start).
        self.gate = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, joint_hidden, *, schedule_scale=1.0, detach=False):
        if joint_hidden.ndim != 3:
            raise ValueError("Expected [B,T,D] joint hidden states")
        source = joint_hidden.detach() if detach else joint_hidden
        delta = self.up(torch.nn.functional.silu(self.down(self.norm(source))))
        return delta * (self.gate * float(schedule_scale))


class ZeroInitJointCrossAttention(nn.Module):
    """Independent AR-to-FM residual with an initially inert output projection."""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        # Keep Q/K/V active, but add exactly zero to the native AR hidden state
        # until the output projection learns. A nonzero gate lets that projection
        # receive gradients on the first step; Q/K/V follow once it is nonzero.
        nn.init.zeros_(self.output.weight)
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, queries, context, *, schedule_scale=1.0, detach_context=False):
        if queries.ndim != 3 or context.ndim != 3 or queries.shape[0] != context.shape[0]:
            raise ValueError("Expected matching [B,T,D] query/context tensors")
        source = context.detach() if detach_context else context
        attended, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(source),
            self.context_norm(source),
            need_weights=False,
        )
        return queries + self.output(attended) * (self.gate * float(schedule_scale))


class PredictedPlanFutureConditioner(nn.Module):
    """Map recovered plan/future x0 values into AR cross-attention tokens."""

    def __init__(self, plan_dim, vision_channels, hidden_size):
        super().__init__()
        self.plan = nn.Linear(plan_dim, hidden_size, bias=False)
        self.future = nn.Linear(vision_channels, hidden_size, bias=False)

    def forward(self, plan_x0, future_x0):
        if plan_x0.ndim != 3 or future_x0.ndim != 5:
            raise ValueError("Expected plan [B,T,D] and future [B,C,T,H,W]")
        # The first vision latent is the observed frame. Only predicted future
        # latents may condition action generation.
        future_tokens = future_x0[:, :, 1:].float().mean(dim=(-1, -2)).transpose(1, 2)
        return torch.cat(
            [
                self.plan(plan_x0.float()),
                self.future(future_tokens),
            ],
            dim=1,
        )


class PredictedFutureConditioner(nn.Module):
    """Map only the predicted future latent into AR cross-attention tokens."""

    def __init__(self, vision_channels, hidden_size):
        super().__init__()
        self.future = nn.Linear(vision_channels, hidden_size, bias=False)

    def forward(self, future_x0):
        if future_x0.ndim != 5 or future_x0.shape[2] < 2:
            raise ValueError("Expected future latents [B,C,T,H,W] with T >= 2")
        future_tokens = future_x0[:, :, 1:].float().mean(dim=(-1, -2)).transpose(1, 2)
        return self.future(future_tokens)


def tokenize_action_targets(tokenizer, texts, *, return_masks=False):
    """Tokenize actions and optionally mask everything outside the official answer JSON."""
    result, masks = [], []
    for text in texts:
        if not isinstance(text, str) or not text:
            raise ValueError("Hybrid AR training requires nonempty action text")
        if return_masks:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(encoded["input_ids"])
            offsets = encoded["offset_mapping"]
            if "<answer>\n" in text and "\n</answer>" in text:
                start = text.index("<answer>\n") + len("<answer>\n")
                end = text.index("\n</answer>", start)
                mask = [token_end > start and token_start < end for token_start, token_end in offsets]
            else:
                mask = [True] * len(ids)
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Tokenizer produced an empty action target")
        eos = tokenizer.eos_token_id
        if eos is not None and ids[-1] != eos:
            ids.append(eos)
            if return_masks:
                mask.append(False if "<answer>" in text else True)
        result.append(ids)
        if return_masks:
            if len(mask) != len(ids) or not any(mask):
                raise ValueError("Action target mask must select at least one token")
            masks.append(mask)
    return (result, masks) if return_masks else result


def autoregressive_ce(logits, labels, *, token_mask=None):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").view_as(
        labels
    )
    mask = torch.ones_like(labels, dtype=torch.bool) if token_mask is None else token_mask.bool()
    if mask.shape != labels.shape or not mask.any():
        raise ValueError("AR token mask must select at least one label")
    return (loss * mask).sum() / mask.sum()


def teacher_kl(student_logits, teacher_logits, *, token_mask=None, temperature=1.0):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Student and teacher logits must match [B,T,V]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_student = torch.nn.functional.log_softmax(student_logits.float() / temperature, -1)
    teacher = torch.nn.functional.softmax(teacher_logits.float() / temperature, -1)
    per_token = torch.nn.functional.kl_div(log_student, teacher, reduction="none").sum(-1)
    mask = torch.ones_like(per_token, dtype=torch.bool) if token_mask is None else token_mask.bool()
    if mask.shape != per_token.shape or not mask.any():
        raise ValueError("KD token mask must select at least one token")
    return temperature**2 * (per_token * mask).sum() / mask.sum()


def recovered_x0(xt, velocity, sigma):
    while sigma.ndim < xt.ndim:
        sigma = sigma.unsqueeze(-1)
    return xt.float() - sigma.float() * velocity.float()


def x0_huber(xt, velocity, sigma, target):
    prediction = recovered_x0(xt, velocity, sigma)
    if prediction.shape != target.shape:
        raise ValueError("Recovered and target x0 shapes differ")
    return torch.nn.functional.huber_loss(prediction, target.float())
