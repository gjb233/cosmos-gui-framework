"""Fixed-layout GUI actions. Text payloads are deliberately not fabricated."""

from dataclasses import dataclass

import torch

from .transition import validate_action

ACTION_TYPES = (
    "click",
    "long_press",
    "input_text",
    "scroll",
    "navigate_home",
    "navigate_back",
    "open_app",
    "wait",
)
DIRECTIONS = ("up", "down", "left", "right")
ACTION_DIM = 14


@dataclass(frozen=True)
class DecodedAction:
    action: dict
    valid: bool
    payload_complete: bool
    error: str | None = None


def encode_action(action: dict) -> torch.Tensor:
    validate_action(action)
    value = torch.zeros(ACTION_DIM, dtype=torch.float32)
    value[:8] = -1
    value[ACTION_TYPES.index(action["type"])] = 1
    if action["type"] in ("click", "long_press"):
        value[8:10] = torch.tensor([action["x"], action["y"]]) * 2 - 1
    if action["type"] == "scroll":
        value[10:14] = -1
        value[10 + DIRECTIONS.index(action["direction"])] = 1
    return value


def applicability(clean_action: torch.Tensor) -> torch.Tensor:
    """Loss-only mask; NEVER pass this as Cosmos action_valid_mask."""
    kinds = clean_action[..., :8].argmax(-1)
    mask = torch.zeros_like(clean_action, dtype=torch.bool)
    mask[..., :8] = True
    mask[..., 8:10] = ((kinds == 0) | (kinds == 1)).unsqueeze(-1)
    mask[..., 10:14] = (kinds == 3).unsqueeze(-1)
    return mask


def decode_action(value: torch.Tensor) -> DecodedAction:
    value = value.detach().float().reshape(-1)
    if value.numel() < ACTION_DIM or not torch.isfinite(value[:ACTION_DIM]).all():
        return DecodedAction({}, False, False, "nonfinite_or_short_action")
    kind = ACTION_TYPES[int(value[:8].argmax())]
    action = {"type": kind}
    if kind in ("click", "long_press"):
        x, y = ((value[8:10] + 1) / 2).tolist()
        action.update(x=x, y=y)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return DecodedAction(action, False, True, "coordinate_out_of_bounds")
    elif kind == "scroll":
        action["direction"] = DIRECTIONS[int(value[10:14].argmax())]
    complete = kind not in ("input_text", "open_app")
    return DecodedAction(action, True, complete, None if complete else "missing_text_payload")
