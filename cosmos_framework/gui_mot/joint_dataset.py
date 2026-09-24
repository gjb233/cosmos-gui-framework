"""H1 GUI samples for the native Cosmos WAM packer."""

import json
import os

import numpy as np
import torch
from PIL import Image

from .action_codec import encode_action
from .hybrid_action import action_trajectory_text, libra_json_trajectory
from .transition import load_manifest


def policy_prompt(instruction, horizon=1, previous_actions=None):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty current instruction is required")
    return json.dumps(
        {
            "task": (
                "Choose the next Android GUI action and predict its resulting screen."
                if horizon == 1
                else f"Predict the next {horizon} Android GUI actions and their resulting screens."
            ),
            "instruction": instruction,
            "previous_actions": previous_actions or [],
            "coordinate_system": "normalized_xy_on_current_image",
        },
        ensure_ascii=False,
    )


def configured_target_hw():
    """Return an optional HxW override shared by training and plan teachers."""
    height = os.environ.get("GUI_TARGET_HEIGHT")
    width = os.environ.get("GUI_TARGET_WIDTH")
    if height is None and width is None:
        return None
    if height is None or width is None:
        raise ValueError("GUI_TARGET_HEIGHT and GUI_TARGET_WIDTH must be set together")
    target_hw = (int(height), int(width))
    if min(target_hw) < 32 or any(size % 32 for size in target_hw):
        raise ValueError("GUI target dimensions must be positive multiples of 32")
    return target_hw


def read_screen(path, target_hw=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if target_hw is None:
            target_hw = configured_target_hw()
        if target_hw is None:
            width, height = image.size
            target_width = ((width + 31) // 32) * 32
            target_height = ((height + 31) // 32) * 32
        else:
            target_height, target_width = target_hw
        if (target_width, target_height) != image.size:
            # Cosmos' VAE requires multiples of 32. A sub-percent resize keeps
            # normalized GUI coordinates invariant while retaining the official
            # screenshot geometry (1080x2400 -> 1088x2400).
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        array = np.array(image)
    height, width = array.shape[:2]
    if height % 32 or width % 32:
        raise ValueError("Screen dimensions must be divisible by 32")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def make_sample(
    current,
    instruction,
    *,
    future=None,
    action=None,
    action_plan=None,
    previous_actions=None,
    max_action_dim=64,
    horizon=1,
    native_ar_text=False,
):
    """Inference may supply ONLY current+instruction; omitted targets are zeros.

    The plan is constructed explicitly: the generic robotics helper mistakes
    T_action=1,T_video=5 for an initial-state action and conditions on it.
    """
    from cosmos_framework.data.generator.action.utils.action_processing import (
        ActionProcessor,
    )
    from cosmos_framework.data.generator.sequence_packing import SequencePlan

    if max_action_dim < 14:
        raise ValueError("max_action_dim must be >=14")
    if current.ndim != 3 or current.shape[0] != 3 or current.dtype != torch.uint8:
        raise ValueError("Expected uint8 CHW current screenshot")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    futures = (
        [torch.zeros_like(current) for _ in range(horizon)]
        if future is None
        else ([future] if isinstance(future, torch.Tensor) else list(future))
    )
    if len(futures) != horizon or any(
        target.shape != current.shape or target.dtype != current.dtype for target in futures
    ):
        raise ValueError("Expected one matching future screen per horizon step")
    height, width = current.shape[-2:]
    if height % 32 or width % 32:
        raise ValueError("Screen dimensions must be divisible by 32")
    actions = [None] * horizon if action is None else ([action] if isinstance(action, dict) else list(action))
    if len(actions) != horizon:
        raise ValueError("Expected one action per horizon step")
    if action_plan is not None:
        value = torch.as_tensor(action_plan, dtype=torch.float32)
        if value.shape != (horizon, max_action_dim) or not torch.isfinite(value).all():
            raise ValueError("Expected finite [horizon,max_action_dim] action plan")
    else:
        value = torch.stack([torch.zeros(14) if item is None else encode_action(item) for item in actions])
    frames = [current]
    for target in futures:
        # Wan's causal VAE compresses every four RGB frames into one future latent.
        frames.extend([target] * 4)
    sample = {
        "video": torch.stack(frames, dim=1),
        "ai_caption": policy_prompt(instruction, horizon, previous_actions),
        "image_size": torch.tensor([height, width, height, width]),
        "padding_mask": torch.zeros(1, height, width),
        "fps": torch.tensor(4.0),
        "conditioning_fps": torch.tensor(4.0),
        "conditioning_fps_action": torch.tensor(1.0),
        "domain_id": torch.tensor(0),
        # The native FM baseline ignores this field. The hybrid recipe uses it
        # as the original GUI-Libra AR/CE target after joint denoising.
        "gui_action_text": (libra_json_trajectory if native_ar_text else action_trajectory_text)(
            [item for item in actions if item is not None]
        )
        if all(item is not None for item in actions)
        else "",
        "sequence_plan": SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=[],
            action_start_frame_offset=1,
        ),
    }
    return ActionProcessor(max_action_dim=max_action_dim).preprocess_action(sample, value, action_normalizer=None)


class JointPolicyDataset:
    def __init__(
        self,
        manifest,
        *,
        split="train",
        max_action_dim=64,
        instruction_level="low",
        horizon=1,
        plan_cache=None,
        native_ar_text=False,
        normalize_plan=False,
    ):
        self.records = load_manifest(manifest)
        if instruction_level not in ("high", "low", "mixed"):
            raise ValueError("instruction_level must be high, low, or mixed")
        if instruction_level in {"high", "mixed"} and any(not r.goal or not r.goal.strip() for r in self.records):
            raise ValueError("High-level training requires episode goals; re-export the manifest")
        if any(row.split != split for row in self.records):
            raise ValueError(f"Manifest contains records outside {split}")
        self.max_action_dim = max_action_dim
        self.instruction_level = instruction_level
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.native_ar_text = bool(native_ar_text)
        self.normalize_plan = bool(normalize_plan)
        self.plan_cache = None
        self.plan_mean = None
        self.plan_std = None
        if plan_cache:
            payload = json.loads(open(plan_cache, encoding="utf-8").read())
            if payload.get("dimension") != max_action_dim:
                raise ValueError("Action-plan cache dimension does not match max_action_dim")
            if self.native_ar_text and payload.get("action_format") != "libra_json_objects":
                raise ValueError("MoT joint training requires native JSON-object teacher plans")
            self.plan_cache = payload.get("windows")
            if not isinstance(self.plan_cache, dict):
                raise ValueError("Action-plan cache is missing windows")
            if self.normalize_plan:
                # NumPy converts the large nested JSON list in native code;
                # torch.as_tensor(list_of_lists) is prohibitively slow here.
                plans = torch.from_numpy(np.asarray(list(self.plan_cache.values()), dtype=np.float32)).reshape(
                    -1, max_action_dim
                )
                if not torch.isfinite(plans).all():
                    raise ValueError("Action-plan cache contains nonfinite values")
                self.plan_mean = plans.mean(dim=0)
                self.plan_std = plans.std(dim=0, unbiased=False).clamp_min(1e-6)
        elif self.normalize_plan:
            raise ValueError("Plan normalization requires an action-plan cache")
        self.windows = []
        for start in range(len(self.records) - horizon + 1):
            window = self.records[start : start + horizon]
            if all(
                right.episode_id == left.episode_id
                and right.step == left.step + 1
                and right.current_image == left.next_image
                for left, right in zip(window, window[1:])
            ):
                self.windows.append(window)
        if not self.windows:
            raise ValueError(f"Manifest has no contiguous {horizon}-step trajectory")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        row = window[0]
        if self.horizon > 1 and self.instruction_level != "high":
            raise ValueError("Multi-step world-action training requires the episode goal")
        use_high = self.instruction_level == "high" or (self.instruction_level == "mixed" and index % 10 < 7)
        instruction = row.goal if use_high else row.instruction
        key = f"{row.episode_id}:{row.step}:{self.horizon}"
        action_plan = None if self.plan_cache is None else self.plan_cache.get(key)
        if self.plan_cache is not None and action_plan is None:
            raise ValueError(f"Action-plan cache is missing {key}")
        if action_plan is not None and self.normalize_plan:
            action_plan = (torch.as_tensor(action_plan, dtype=torch.float32) - self.plan_mean) / self.plan_std
        current = read_screen(row.current_image)
        target_hw = tuple(current.shape[-2:])
        return make_sample(
            current,
            instruction,
            future=[read_screen(item.next_image, target_hw) for item in window],
            action=[item.action for item in window],
            action_plan=action_plan,
            previous_actions=row.previous_actions,
            max_action_dim=self.max_action_dim,
            horizon=self.horizon,
            native_ar_text=self.native_ar_text,
        )


class StreamingJointPolicyDataset(torch.utils.data.IterableDataset):
    """Infinite deterministic stream for the native packer, partitioned by rank/worker."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dataset = JointPolicyDataset(*args, **kwargs)
        self.shard_rank = 0
        self.shard_world_size = 1

    def __len__(self):
        # Native PackingDataLoader requires a length even for infinite streams.
        # Match its iterable-dataset convention; max_iter ends training.
        return 10**12

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        index = self.shard_rank * workers + worker_id
        stride = self.shard_world_size * workers
        if stride < 1 or not 0 <= self.shard_rank < self.shard_world_size:
            raise ValueError("Invalid GUI training stream shard")
        while True:
            yield self.dataset[index % len(self.dataset)]
            index += stride
