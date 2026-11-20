"""Explicit native frontend/head boundary for the distributed driver.
No detaching/copying/offloading activations. Rank placement and loss are external.
"""
from split_network import build_split_forward
from native_frontend import prepare_native_ar,prepare_fm_decoder,finish_native_ar,finish_fm_decoder

class NativeEndpoints:
    def __init__(self, network):
        self.network = network
        self.prepare_fm,self.finish_fm,_ = build_split_forward(type(network))
        self.pending = {}

    def prepare(self, key, packed_seq, memory=None, video_temporal_causal=None,
                correct_cp_gradients=False):
        if key in self.pending:
            raise ValueError("Duplicate live endpoint microbatch")
        if not getattr(packed_seq,"gui_native_fm_ar",False):
            raise ValueError("Native FM/AR required")
        context=self.prepare_fm(self.network,packed_seq,memory,
                                video_temporal_causal,correct_cp_gradients)
        from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
        ar=prepare_native_ar(self.network,packed_seq,get_gen_seq(context["input_pack"]))
        if ar is None:
            raise ValueError("Missing AR supervision")
        pack,rope,visual=prepare_fm_decoder(self.network.language_model.model,
                         context["input_pack"],context["packed_position_ids"])
        self.pending[key]=(context,ar)
        return dict(fm_pack=pack,fm_position_embeddings=rope,ar_hidden=ar["ar_hidden"],
                    ar_cos=ar["ar_cos"],ar_sin=ar["ar_sin"],
                    ar_visual_mask=ar["ar_visual_mask"],ar_deepstack=ar["ar_deepstack"],
                    fm_visual=visual,attention_mask=context["attention_meta"])

    def head_context(self, key):
        """Select the original FM head arguments and teacher-forced AR labels."""
        context, ar = self.pending[key]
        fm_fields = ("packed_seq", "original_latent_shapes", "original_latent_shapes_lidar",
                     "sequence_shard_parallel_dims", "correct_cp_gradients")
        ar_fields = ("labels", "token_mask", "prompt_length", "weights")
        return ({name: context[name] for name in fm_fields},
                {name: ar[name] for name in ar_fields})

    def finish_local_heads(self, decoded_pack, decoded_ar, head_context):
        """Run on the final stage using that rank's head weights and local context."""
        fm_context, ar_context = head_context
        pack = finish_fm_decoder(self.network.language_model.model, decoded_pack)
        output = self.finish_fm(self.network, pack, fm_context)
        output.update(finish_native_ar(self.network, decoded_ar, ar_context))
        return output

    def finish(self, key, decoded_pack, decoded_ar):
        return self.finish_local_heads(decoded_pack, decoded_ar, self.head_context(key))

    def release_after_backward(self, key):
        # The caller owns the graph and must finish backward before releasing.
        del self.pending[key]
