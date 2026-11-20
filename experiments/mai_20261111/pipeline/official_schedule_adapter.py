"""Official schedule integration; full model rank construction remains external.
Only floating activation wires cross PipelineStage. PyTorch 2.10 marks every
received input requires_grad, so an int64 ID cannot be a pipeline activation.
Keys are pre-registered in identical forward order on each rank.
"""
import torch
from torch import nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

class OrderedJointStage(nn.Module):
    def __init__(self, flat_stage):
        super().__init__()
        self.flat_stage = flat_stage
        self.keys = ()
        self.cursor = 0
        self.active = False

    def begin_step(self, keys):
        if self.active:
            raise RuntimeError("Previous scheduled step has not finished backward")
        keys = tuple(keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Need nonempty unique microbatch keys")
        if any(key not in self.flat_stage.contexts for key in keys):
            raise ValueError("Register local metadata before beginning a schedule")
        self.keys, self.cursor, self.active = keys, 0, True

    def forward(self, causal, full, ar):
        if not self.active or self.cursor >= len(self.keys):
            raise RuntimeError("Unexpected forward; recomputation is not enabled")
        key = self.keys[self.cursor]
        self.cursor += 1
        # Local control only: this integer never travels in PipelineStage wires.
        token = torch.tensor([key], dtype=torch.int64, device=causal.device)
        return self.flat_stage(causal, full, ar, token)[:3]

    def finish_step(self):
        # Invoke only after schedule.step has returned, including all backward.
        if not self.active or self.cursor != len(self.keys):
            raise RuntimeError("Not all microbatches were forwarded")
        self.flat_stage.clear_completed_step()
        self.keys, self.cursor, self.active = (), 0, False

def build_decoder_schedule(module, stage_index, num_stages, device, capacities,
                           dtypes, n_microbatches, loss_fn=None, group=None,
                           schedule_kind="1f1b"):
    if len(capacities) != 3 or len(dtypes) != 3:
        raise ValueError("Three activation wires required")
    if schedule_kind not in ("1f1b", "gpipe"):
        raise ValueError(schedule_kind)
    if n_microbatches < 1:
        raise ValueError(n_microbatches)
    examples = tuple(torch.empty(capacity, dtype=dtype, device=device)
                     for capacity, dtype in zip(capacities, dtypes))
    if any(not tensor.is_floating_point() for tensor in examples):
        raise ValueError("Pipeline activation wires must be floating point")
    stage = PipelineStage(module, stage_index, num_stages, device,
                          input_args=examples, output_args=examples, group=group)
    schedule_cls = Schedule1F1B if schedule_kind == "1f1b" else ScheduleGPipe
    return schedule_cls(stage, n_microbatches=n_microbatches,
                        loss_fn=loss_fn, scale_grads=True)
