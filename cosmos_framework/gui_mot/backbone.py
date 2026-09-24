"""Resolve a single GUI reasoner for both weight loading and preprocessing."""

import os
from pathlib import Path


def backbone_path(environ=None):
    env = os.environ if environ is None else environ
    path = env.get("GUI_BACKBONE_PATH") or env.get("MAI_UI_PATH")
    if not path:
        raise ValueError("Set GUI_BACKBONE_PATH to a local Qwen3-VL GUI checkpoint")
    return path


def configure_backbone(*, require_exists=True):
    path = backbone_path()
    if require_exists:
        root = Path(path)
        for filename in (
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        ):
            if not (root / filename).is_file():
                raise ValueError(f"GUI backbone asset missing: {root / filename}")
    os.environ["GUI_BACKBONE_PATH"] = path
    return path
