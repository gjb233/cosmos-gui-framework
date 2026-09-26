"""AndroidControl transition adapter for the native Cosmos action FM path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.sequence_packing import SequencePlan

ACTION_TYPES = (
    "click",
    "long_press",
    "scroll",
    "navigate_home",
    "navigate_back",
    "wait",
)
DIRECTIONS = ("up", "down", "left", "right")
UNSUPPORTED_TEXT_TYPES = {"input_text", "open_app"}


def encode_gui_action(action: dict) -> torch.Tensor:
    """12-D model-space action covering the six payload-free GUI types."""
    kind = action["type"]
    if kind not in ACTION_TYPES:
        raise ValueError(f"Unknown GUI action: {kind}")
    value = torch.zeros(12, dtype=torch.float32)
    value[:6] = -1
    value[ACTION_TYPES.index(kind)] = 1
    if kind in ("click", "long_press"):
        x, y = float(action["x"]), float(action["y"])
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"Invalid click position: {(x, y)}")
        value[6:8] = torch.tensor([2 * x - 1, 2 * y - 1])
    if kind == "scroll":
        value[8:12] = -1
        value[8 + DIRECTIONS.index(action["direction"])] = 1
    return value


def decode_gui_action(value: torch.Tensor) -> dict:
    value = value.detach().float().reshape(-1)
    if value.numel() < 12 or not torch.isfinite(value[:12]).all():
        raise ValueError("GUI action output is short or nonfinite")
    kind = ACTION_TYPES[int(value[:6].argmax())]
    action = {"type": kind}
    if kind in ("click", "long_press"):
        action.update(zip(("x", "y"), ((value[6:8].clamp(-1, 1) + 1) / 2).tolist()))
    if kind == "scroll":
        action["direction"] = DIRECTIONS[int(value[8:12].argmax())]
    return action


def read_screen(path: str, width: int = 160, height: int = 352) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


class GUITransitionDataset(Dataset):
    def __init__(self, manifest: str, max_action_dim: int = 64, tokenizer_config: dict | None = None):
        manifest_path = Path(manifest).resolve()
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.action_counts = {
            kind: sum(row["action"]["type"] == kind for row in rows)
            for kind in (*ACTION_TYPES, *UNSUPPORTED_TEXT_TYPES)
        }
        self.rows = [row for row in rows if row["action"]["type"] not in UNSUPPORTED_TEXT_TYPES]
        if not self.rows:
            raise ValueError("Empty GUI manifest")
        if any(row.get("split") != "train" for row in self.rows):
            raise ValueError("GUI manifest must contain only training rows")
        self.transform = ActionTransformPipeline(
            pad_keys=[],
            tokenizer_config=tokenizer_config,
            max_action_dim=max_action_dim,
            cfg_dropout_rate=0.1,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        current = read_screen(row["current_image"])
        future = read_screen(row["next_image"])
        video = torch.stack([current, future, future, future, future], dim=1)
        sample = {
            "video": video,
            "action": encode_gui_action(row["action"]).unsqueeze(0),
            "ai_caption": json.dumps(
                {"instruction": row["instruction"], "task": "Predict the next GUI action and resulting screen."},
                ensure_ascii=False,
            ),
            "mode": "wam",
            "fps": torch.tensor(4.0),
            "conditioning_fps": torch.tensor(4.0),
            "conditioning_fps_action": torch.tensor(1.0),
            "domain_id": torch.tensor(33),  # dedicated Android GUI action domain
        }
        sample = self.transform(sample, resolution="256")
        sample["image_size"] = torch.tensor([352, 160, 352, 160], dtype=torch.float32)
        # The generic length heuristic treats a single action with five RGB
        # frames as an observed initial-state action. It must be predicted.
        sample["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=[],
            action_start_frame_offset=1,
        )
        sample.pop("mode", None)
        return sample


class ShardedGUITransitionDataset(IterableDataset):
    def __init__(self, dataset: GUITransitionDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        shard_count = self.shard_world_size * worker_count
        shard_id = self.shard_rank * worker_count + worker_id
        epoch = 0
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.dataset), generator=generator).tolist()
            for index in order[shard_id::shard_count]:
                yield self.dataset[index]
            epoch += 1


def get_gui_transition_dataset(
    manifest: str, max_action_dim: int = 64, tokenizer_config: dict | None = None
) -> ShardedGUITransitionDataset:
    dataset = GUITransitionDataset(manifest, max_action_dim, tokenizer_config)
    return ShardedGUITransitionDataset(dataset)
