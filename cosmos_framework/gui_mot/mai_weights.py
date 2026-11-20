"""Stream Qwen3-VL GUI language + visual tensors into Cosmos without touching DM weights."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def gui_key(name, *, copy_generator=False):
    name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    if ".lora_" in name:
        return None
    if "moe_gen" in name:
        if not copy_generator:
            return None
        name = name.replace("_moe_gen", "")
    if name.startswith("visual."):
        return "model." + name
    if name.startswith("model."):
        return "model.language_model." + name[6:]
    if name == "lm_head.weight":
        return name
    raise ValueError(f"Unrecognized reasoner parameter {name}")


@torch.no_grad()
def load_gui_reasoner(language_model, directory, *, audit_only=False, copy_generator=False):
    from torch.distributed.tensor import DTensor, distribute_tensor

    root = Path(directory)
    index = root / "model.safetensors.index.json"
    if index.exists():
        files = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
            files = {key: "model.safetensors" for key in handle.keys()}  # noqa: SIM118 -- safe_open is not a dict
    mapped = []
    # Audit complete shapes before writing the first parameter.
    for name, parameter in language_model.named_parameters():
        source = gui_key(name, copy_generator=copy_generator)
        if source is None:
            continue
        if source not in files:
            if copy_generator and "moe_gen" in name and ("q_norm_moe_gen" in name or "k_norm_moe_gen" in name):
                continue
            raise ValueError(f"GUI backbone tensor missing: {source}")
        with safe_open(root / files[source], framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(source).get_shape())
        if shape != tuple(parameter.shape):
            raise ValueError(f"GUI backbone shape mismatch: {source}: {shape} vs {tuple(parameter.shape)}")
        mapped.append((parameter, source, files[source]))
    if not any(source.startswith("model.visual.") for _, source, _ in mapped):
        raise ValueError("Model has no visual tower; include_visual must be enabled")
    if audit_only:
        return len(mapped)
    for parameter, source, filename in mapped:
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(source).to(device=parameter.device, dtype=parameter.dtype)
        if isinstance(parameter, DTensor):
            value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
        parameter.copy_(value)
    return len(mapped)


# Backwards compatibility for existing MAI callers.
mai_key = gui_key
load_mai_reasoner = load_gui_reasoner
