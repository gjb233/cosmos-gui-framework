"""Run the native Cosmos trainer with the cluster's DTensor init workaround."""

import os
import runpy


def main() -> None:
    if os.environ.get("GUI_DTENSOR_SAFE_INIT") == "1":
        import torch
        from torch.distributed.tensor import DTensor

        original = torch.nn.init.trunc_normal_

        def safe_trunc_normal(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
            if not isinstance(tensor, DTensor):
                return original(tensor, mean, std, a, b, generator=generator)
            with torch.no_grad():
                tensor.to_local().normal_(mean, std, generator=generator).clamp_(a, b)
            return tensor

        torch.nn.init.trunc_normal_ = safe_trunc_normal

    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
