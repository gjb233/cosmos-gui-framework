"""Per-microbatch detached metadata transport; activations use PipelineStage.
Control schemas travel over a Gloo group; tensors stay on GPU via NCCL.
Every rank must call exchange in the same order. No backward data travels here.
"""
import torch
import torch.distributed as dist
from boundary_codec import BoundaryCodec

class MicrobatchMetadataChannel:
    def __init__(self, control_group, tensor_group, device):
        self.control_group = control_group
        self.tensor_group = tensor_group
        self.device = torch.device(device)
        self.seen = set()

    def exchange(self, key, value=None, source=0):
        rank = dist.get_rank()
        error = None
        codec = None
        tensors = ()
        if key in self.seen:
            error = "Duplicate microbatch key"
        if rank == source:
            try:
                codec, tensors = BoundaryCodec.flatten(value)
                for tensor in tensors:
                    if tensor.requires_grad:
                        raise ValueError("Trainable features must be explicit pipeline inputs")
                    if tensor.device != self.device or tensor.device.type != "cuda":
                        raise ValueError("Metadata tensors must be local CUDA tensors")
                    if tensor.layout != torch.strided:
                        raise ValueError("Only strided metadata tensors are supported")
            except Exception as exc:
                error = repr(exc)
        # Propagate validation errors before any rank enters NCCL collectives.
        errors = [None] * dist.get_world_size(self.control_group)
        dist.all_gather_object(errors, error, group=self.control_group)
        if any(item is not None for item in errors):
            raise ValueError("Metadata preflight failed: " + repr(errors))
        header = [(key, codec.schema, [(tuple(t.shape), t.dtype) for t in tensors])
                  if rank == source else None]
        dist.broadcast_object_list(header, src=source, group=self.control_group)
        received_key, schema, specs = header[0]
        mismatches = [None] * dist.get_world_size(self.control_group)
        dist.all_gather_object(mismatches, received_key != key, group=self.control_group)
        if any(mismatches):
            raise ValueError("Ranks supplied different microbatch keys")
        received = []
        for index, (shape, dtype) in enumerate(specs):
            tensor = (tensors[index].detach().contiguous().clone() if rank == source
                      else torch.empty(shape, dtype=dtype, device=self.device))
            dist.broadcast(tensor, src=source, group=self.tensor_group)
            received.append(tensor)
        self.seen.add(key)
        return BoundaryCodec(schema).rebuild(tuple(received), device=self.device)

    def clear_completed_step(self):
        # Caller must finish all scheduled backwards before releasing contexts.
        self.seen.clear()
