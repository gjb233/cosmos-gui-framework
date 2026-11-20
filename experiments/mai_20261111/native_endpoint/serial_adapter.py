"""Serial integration oracle before enabling distributed scheduling.
Uses existing four-card placement hooks. This is NOT a pipeline scheduler.
"""
import torch
from joint_stage import JointDecoderStage
from split_network import build_split_forward
from native_frontend import (prepare_native_ar,prepare_fm_decoder,
                             finish_native_ar,finish_fm_decoder)

def build_serial_adapter(network):
    prepare_fm,finish_fm,_ = build_split_forward(type(network))
    layers=network.language_model.model.layers
    bounds=(0, 3, 9, 15, 21, 26, 31, 36)
    stages=[JointDecoderStage(layers,bounds[i],bounds[i+1]) for i in range(7)]
    def forward(packed_seq,memory=None,video_temporal_causal=None,correct_cp_gradients=False):
        if not getattr(packed_seq,'gui_native_fm_ar',False):
            raise ValueError('Native FM/AR required')
        context=prepare_fm(network,packed_seq,memory,video_temporal_causal,correct_cp_gradients)
        from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
        preparation=prepare_native_ar(network,packed_seq,get_gen_seq(context['input_pack']))
        if preparation is None:
            raise ValueError('Missing native AR supervision')
        pack,rope,visual=prepare_fm_decoder(network.language_model.model,
                                             context['input_pack'],context['packed_position_ids'])
        ar=preparation['ar_hidden']
        for stage in stages:
            pack,ar=stage(pack,rope,ar,preparation['ar_cos'],preparation['ar_sin'],
                          preparation['ar_visual_mask'],preparation['ar_deepstack'],
                          fm_visual=visual,attention_mask=context['attention_meta'])
        pack=finish_fm_decoder(network.language_model.model,pack)
        output=finish_fm(network,pack,context)
        output.update(finish_native_ar(network,ar,preparation))
        return output
    return forward
