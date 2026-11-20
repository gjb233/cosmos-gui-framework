"""Atomic AR-only exports and finite-value monitoring for native GUI runs."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class NativeTrainingArtifacts(Callback):
    def __init__(self, output, every_n=250):
        super().__init__()
        self.output = Path(output)
        self.every_n = int(every_n)
        self.last_export = None

    @torch.no_grad()
    def export(self, model, iteration):
        if self.last_export == iteration:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"ar-step-{iteration:06d}.safetensors"
        state = {}
        for name, parameter in model.net.named_parameters():
            if "lora_" not in name or "_moe_gen" in name:
                continue
            value = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
            if dist.get_rank() == 0:
                state[name] = value.detach().cpu().clone()
        if dist.get_rank() == 0:
            from safetensors.torch import save_file

            if len(state) != 288 or any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Invalid AR adapter export")
            temporary = path.with_suffix(".tmp")
            save_file(state, str(temporary))
            temporary.replace(path)
            path.with_suffix(".ready.json").write_text(json.dumps({"iteration": iteration, "tensors": len(state)}))
        dist.barrier()
        self.last_export = iteration

    def on_train_start(self, model, iteration=0):
        self.freeze_reference = None
        if os.environ.get("GUI_FREEZE_FM_LORA") == "1":
            fm = {n: p for n, p in model.net.named_parameters() if "lora_" in n and "_moe_gen" in n}
            ar = {n: p for n, p in model.net.named_parameters() if "lora_" in n and "_moe_gen" not in n}
            assert len(fm) == len(ar) == 288
            assert all(not p.requires_grad for p in fm.values())
            assert all(p.requires_grad for p in ar.values())
            self.freeze_reference = {
                n: (p.to_local() if hasattr(p, "to_local") else p).detach().cpu().clone() for n, p in fm.items()
            }
            if dist.get_rank() == 0:
                (self.output / "freeze-policy.json").parent.mkdir(parents=True, exist_ok=True)
                (self.output / "freeze-policy.json").write_text(
                    json.dumps({"fm_lora_frozen_tensors": len(fm), "ar_lora_trainable_tensors": len(ar)})
                )
        self.export(model, iteration)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        finite = torch.isfinite(loss.detach()).all().to(device="cuda", dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite training loss on at least one rank")
        if dist.get_rank() == 0:
            record = {
                "iteration": iteration,
                "loss_rank0_last_microbatch": float(loss.detach()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            # Keep local progress aligned with the scalar training objectives.
            for name in ("gui_ar_ce", "gui_ar_kd", "flow_matching_loss_vision"):
                if name in output_batch:
                    record[f"{name}_rank0_last_microbatch"] = float(output_batch[name].detach())
            with (self.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if self.freeze_reference is not None and (iteration <= 3 or iteration % self.every_n == 0 or iteration == 100):
            ar_b_sum = torch.zeros((), device="cuda", dtype=torch.float64)
            for name, parameter in model.net.named_parameters():
                local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                if name in self.freeze_reference:
                    assert not parameter.requires_grad
                    assert torch.equal(local.detach().cpu(), self.freeze_reference[name]), name
                if "lora_B" in name and "_moe_gen" not in name:
                    ar_b_sum += local.detach().double().abs().sum()
            dist.all_reduce(ar_b_sum)
            if iteration >= 2 and ar_b_sum.item() == 0:
                raise RuntimeError("AR LoRA B did not update after warmup started")
            if dist.get_rank() == 0:
                with (self.output / "freeze-audit.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"iteration": iteration, "frozen_fm_unchanged": True, "ar_lora_b_abs_sum": ar_b_sum.item()}
                        )
                        + "\n"
                    )
        if iteration % self.every_n == 0 or iteration == 100:
            self.export(model, iteration)

    def on_train_end(self, model, iteration=0):
        self.export(model, iteration)
