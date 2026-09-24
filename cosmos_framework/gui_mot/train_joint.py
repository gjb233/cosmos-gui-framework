"""Thin launcher: custom registration followed by the native Cosmos trainer."""

import os
import runpy
import sys
from pathlib import Path


def _install_dtensor_init_workaround():
    """Avoid a CUDA 13.2 DTensor erfinv driver failure on the cluster.

    Cosmos immediately replaces these base parameters from its DCP checkpoint.
    Keep this opt-in so normal training environments retain PyTorch's exact
    truncated-normal initializer.
    """
    if os.environ.get("GUI_DTENSOR_SAFE_INIT") != "1":
        return
    import torch
    from torch.distributed.tensor import DTensor

    original = torch.nn.init.trunc_normal_

    def safe_trunc_normal(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
        if not isinstance(tensor, DTensor):
            return original(tensor, mean, std, a, b, generator=generator)
        with torch.no_grad():
            local = tensor.to_local()
            local.normal_(mean, std, generator=generator).clamp_(a, b)
        return tensor

    torch.nn.init.trunc_normal_ = safe_trunc_normal


def main():
    from .backbone import configure_backbone

    configure_backbone(require_exists="--dryrun" not in sys.argv)
    _install_dtensor_init_workaround()
    if "--dryrun" not in sys.argv:
        for key in (
            "GUI_BACKBONE_PATH",
            "GUI_TRAIN_MANIFEST",
            "WAN_VAE_PATH",
            "BASE_CHECKPOINT_PATH",
        ):
            if not os.environ.get(key) or not Path(os.environ[key]).exists():
                raise ValueError(f"Set {key} to an existing local asset before training")
    from .experiment import register

    register()
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
