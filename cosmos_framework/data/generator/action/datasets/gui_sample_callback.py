"""One-sample Cosmos-native action generation check for GUI DCP checkpoints."""

import json
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.action.datasets.gui_transition_dataset import (
    GUITransitionDataset,
    decode_gui_action,
)
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.utils.callback import Callback


class GUISampleCallback(Callback):
    def __init__(self, manifest: str, output: str, num_steps: int = 4):
        super().__init__()
        self.manifest = manifest
        self.output = Path(output)
        self.num_steps = num_steps

    @torch.no_grad()
    def on_train_start(self, model, iteration=0):
        row = GUITransitionDataset(self.manifest)[0]
        # Sampling must see only the current screenshot and instruction.
        row["video"][:, 1:] = 0
        row["action"].zero_()
        row["action_raw"].zero_()
        batch = custom_collate_fn([row])
        was_training = model.training
        model.eval()
        try:
            generated = model.generate_samples_from_batch(
                batch, guidance=1.0, seed=[42], n_sample=1, num_steps=self.num_steps
            )
        finally:
            model.train(was_training)
        action = generated["action"][0].detach().float().cpu().reshape(-1)
        result = {
            "iteration": iteration,
            "num_steps": self.num_steps,
            "raw_action": action[:12].tolist(),
            "finite": bool(torch.isfinite(action[:12]).all()),
            "decoded_action": None,
        }
        try:
            result["decoded_action"] = decode_gui_action(action)
        except ValueError as error:
            result["decode_error"] = str(error)
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"GUI_SAMPLE_RESULT {json.dumps(result)}", flush=True)
