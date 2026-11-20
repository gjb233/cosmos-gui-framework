"""Tensor-only adapter for torch.distributed.pipelining PipelineStage.
The caller must register the same microbatch IDs and local metadata on every
rank before a schedule step. Frontend, heads and loss stay outside this adapter.
"""
import torch
from torch import nn
from transport_buffers import TensorEnvelope

class FlatJointPipelineStage(nn.Module):
    def __init__(self, decoder, capacities):
        super().__init__()
        self.decoder = decoder
        self.capacities = tuple(capacities)
        if len(self.capacities) != 3:
            raise ValueError("Need causal FM, full FM and AR capacities")
        self.contexts = {}

    def register_microbatch(self, key, *, fm_pack, fm_position_embeddings,
                            ar_shape, ar_dtype, ar_cos, ar_sin,
                            ar_visual_mask, ar_deepstack, fm_visual, attention_mask):
        if key in self.contexts:
            raise ValueError("Duplicate microbatch key")
        # Do not retain the gradient-carrying inputs in the context cache.
        context_pack = {k:v for k,v in fm_pack.items()
                        if k not in ('causal_seq','full_only_seq')}
        shapes = [fm_pack['causal_seq'].shape,fm_pack['full_only_seq'].shape,ar_shape]
        dtypes = [fm_pack['causal_seq'].dtype,fm_pack['full_only_seq'].dtype,ar_dtype]
        envelope = TensorEnvelope(shapes,dtypes,self.capacities)
        context = dict(fm_position_embeddings=fm_position_embeddings,
                       ar_cos=ar_cos,ar_sin=ar_sin,ar_visual_mask=ar_visual_mask,
                       ar_deepstack=ar_deepstack,fm_visual=fm_visual,
                       attention_mask=attention_mask)
        # Only detached model metadata belongs here. Trainable features must
        # travel as differentiable stage inputs, never hidden in a side channel.
        from boundary_codec import BoundaryCodec
        _, tensors = BoundaryCodec.flatten((context_pack,context))
        device = next(self.decoder.parameters()).device
        for tensor in tensors:
            if tensor.requires_grad:
                raise ValueError("Gradient-carrying metadata requires an explicit pipeline input")
            if tensor.device != device:
                raise ValueError("Stage metadata must already reside on the local device")
        self.contexts[key] = (context_pack,context,envelope)
        return envelope

    def forward(self, fm_causal_wire, fm_full_wire, ar_wire, microbatch_id):
        if microbatch_id.dtype != torch.int64 or tuple(microbatch_id.shape) != (1,):
            raise ValueError("Expected one int64 microbatch ID")
        key = int(microbatch_id.item())  # control scalar only, never activation offload
        pack_template,context,envelope = self.contexts[key]
        causal,full,ar = envelope.unpack((fm_causal_wire,fm_full_wire,ar_wire))
        pack = dict(pack_template,causal_seq=causal,full_only_seq=full)
        out_pack,out_ar = self.decoder(pack,ar_hidden=ar,**context)
        wires = envelope.pack((out_pack['causal_seq'],out_pack['full_only_seq'],out_ar))
        return (*wires,microbatch_id.clone())

    def clear_completed_step(self):
        # Call only after schedule backward has finished for all microbatches.
        self.contexts.clear()
