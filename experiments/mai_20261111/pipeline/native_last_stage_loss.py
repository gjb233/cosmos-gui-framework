"""Final-rank MAI heads and unchanged mixed-source training loss."""
import torch
import torch.distributed as dist

from native_endpoints import NativeEndpoints


class NativeLastStageLoss:
    def __init__(self, model, data_parallel_group):
        if data_parallel_group is None or dist.get_world_size(data_parallel_group) != 1:
            raise ValueError("Seven-stage reference uses one data replica")
        if dist.get_rank() != 6 or dist.get_rank(data_parallel_group) != 0:
            raise ValueError("Expected a singleton group containing the last stage")
        if "mixed_mask" not in model._compute_losses.__func__.__globals__:
            raise ValueError("Install the serial mixed-source FM masking policy first")
        self.model = model
        self.endpoints = NativeEndpoints(model.net)
        self.group = data_parallel_group

    def __call__(self, decoded_pack, decoded_ar, head_context, *,
                 data_batch_packed, gen_data_noised, timesteps, is_image_batch, fm_mask):
        if fm_mask not in (0, 1):
            raise ValueError("Expected the verified transition flag for one sample")
        model = self.model
        fields = ("_loss_averaging_group", "_mixed_fm_mask")
        saved = {field: vars(model)[field] for field in fields if field in vars(model)}
        try:
            model._loss_averaging_group = lambda: (self.group, 1)
            model._mixed_fm_mask = [fm_mask]
            output = self.endpoints.finish_local_heads(decoded_pack, decoded_ar, head_context)
            loss, metrics = model._compute_losses(output, data_batch_packed,
                                                   gen_data_noised, timesteps, is_image_batch)
            if loss.ndim or not loss.requires_grad:
                raise RuntimeError("Expected a differentiable scalar MAI objective")
            return loss, metrics
        finally:
            for field in fields:
                if field in saved:
                    setattr(model, field, saved[field])
                else:
                    delattr(model, field)
