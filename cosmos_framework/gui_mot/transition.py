"""One observed screen and one specified action predict one subsequent screen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

ACTION_FIELDS = {
    "click": {"x", "y"},
    "long_press": {"x", "y"},
    "scroll": {"direction"},
    "input_text": {"text"},
    "open_app": {"app_name"},
    "navigate_home": set(),
    "navigate_back": set(),
    "wait": set(),
}


def validate_action(action):
    kind = action.get("type")
    if kind not in ACTION_FIELDS:
        raise ValueError(f"Unsupported action type: {kind}")
    if set(action) != {"type"} | ACTION_FIELDS[kind]:
        raise ValueError(f"Unexpected/missing fields for {kind}: {set(action)}")
    if kind in {"click", "long_press"}:
        for key in ("x", "y"):
            value = action[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a normalized number")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} outside [0,1]")
    if kind == "scroll" and action["direction"] not in {"up", "down", "left", "right"}:
        raise ValueError("Invalid scroll direction")
    for key in ("text", "app_name"):
        if key in action and (not isinstance(action[key], str) or not action[key]):
            raise ValueError(f"{key} must be a nonempty string")


@dataclass(frozen=True)
class Transition:
    episode_id: str
    step: int
    split: str
    current_image: str
    next_image: str
    instruction: str
    action: dict
    goal: str | None = None
    previous_actions: list[dict] | None = None
    bbox_label: list[float] | None = None  # Evaluation only: x1,y1,x2,y2,valid.

    def __post_init__(self):
        validate_action(self.action)
        if self.bbox_label is not None:
            if len(self.bbox_label) != 5 or not all(math.isfinite(float(v)) for v in self.bbox_label):
                raise ValueError("Expected finite [x1,y1,x2,y2,valid] bbox label")
            x1, y1, x2, y2, valid = self.bbox_label
            if valid > 0.5 and (x1 > x2 or y1 > y2):
                raise ValueError("Inverted target bbox")
        if not self.episode_id or type(self.step) is not int or self.step < 0:
            raise ValueError("Invalid episode/step")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Instruction must be nonempty")
        if self.previous_actions is not None:
            if not isinstance(self.previous_actions, list):
                raise ValueError("previous_actions must be a list")
            for previous in self.previous_actions:
                validate_action(previous)
        if not self.current_image or not self.next_image:
            raise ValueError("Both images are required")

    def prompt(self):
        # Whitelist only current-step information. No goal, later instructions,
        # target captions, bounding-box labels, or target image paths.
        return json.dumps(
            {
                "task": "Predict the screen after executing this Android GUI action.",
                "instruction": self.instruction,
                "coordinate_system": "normalized_xy_on_current_image",
                "action": self.action,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def messages(self):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.current_image},
                    {"type": "text", "text": self.prompt()},
                ],
            }
        ]


def load_manifest(path):
    path = Path(path).resolve()
    records, keys, episode_splits = [], set(), {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record = Transition(**raw)
            key = (record.episode_id, record.step)
            if key in keys:
                raise ValueError(f"Duplicate transition: {key}")
            if episode_splits.setdefault(record.episode_id, record.split) != record.split:
                raise ValueError("Episode crosses dataset splits")
            keys.add(key)
            for field in ("current_image", "next_image"):
                image = (path.parent / raw[field]).resolve()
                if not image.is_file():
                    raise ValueError(f"Missing image: {image}")
                raw[field] = str(image)
            records.append(Transition(**raw))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("Empty manifest")
    return records
