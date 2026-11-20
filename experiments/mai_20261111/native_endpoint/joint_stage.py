"""Real MoT decoder-stage adapter; transport integration is separate."""
import torch
from torch import nn

class JointDecoderStage(nn.Module):
    def __init__(self, layers, start, stop):
        super().__init__()
        if not 0 <= start < stop <= len(layers):
            raise ValueError((start, stop, len(layers)))
        # One set of layer objects serves both forwards; never clone AR weights.
        self.layers = nn.ModuleList(list(layers[start:stop]))
        self.start = start

    def forward(self, fm_pack, fm_position_embeddings, ar_hidden, ar_cos, ar_sin,
                ar_visual_mask, ar_deepstack, fm_visual=None, attention_mask=None):
        from cosmos_framework.gui_mot.multimodal import add_deepstack
        if ar_deepstack and ar_visual_mask is None:
            raise ValueError("Visual mask required for AR deepstack")
        for local_index, layer in enumerate(self.layers):
            layer_index = self.start + local_index
            fm_pack, auxiliary, kv = layer(
                fm_pack, attention_mask, fm_position_embeddings,
                natten_metadata=None, memory_value=None, gen_only=False,
            )
            if auxiliary:
                raise NotImplementedError("Auxiliary losses must be wired before supporting MoE")
            if kv is not None:
                raise ValueError("Unexpected memory output in dense native training")
            if fm_visual is not None:
                add_deepstack(fm_pack, fm_visual, layer_index)
            ar_hidden = layer.reasoner_forward(ar_hidden, ar_cos, ar_sin, None, layer_index)
            if layer_index < len(ar_deepstack):
                ar_hidden = ar_hidden.clone()
                ar_hidden[ar_visual_mask, :] = (
                    ar_hidden[ar_visual_mask, :] + ar_deepstack[layer_index]
                )
        return fm_pack, ar_hidden
