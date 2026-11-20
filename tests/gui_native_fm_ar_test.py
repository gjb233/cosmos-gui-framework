"""CPU regressions: native mask, FM-to-AR gradients and AR-only deployment.

Run: python -m unittest discover -s tests -p gui_native_fm_ar_test.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.runtime import sequence_pack_from_packed_sequence
from cosmos_framework.gui_mot.attention import gui_two_way_sdpa
from cosmos_framework.gui_mot.joint_policy import JointPolicy
from cosmos_framework.gui_mot.multimodal import hybrid_ar_forward


def pack(x):
    return sequence_pack_from_packed_sequence(
        x,
        ["causal", "full", "causal", "full"],
        [3, 2, 3, 2],
        [5, 5],
        torch.tensor([0, 1, 2, 5, 6, 7]),
        torch.tensor([3, 4, 8, 9]),
    )


class NativeAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(10, 2, 4)
        self.k = torch.randn(10, 2, 4)
        self.v = torch.randn(10, 2, 4)
        self.mask = SimpleNamespace(is_three_way=False)

    def run_attention(self, q, k, v):
        return gui_two_way_sdpa(pack(q), pack(k), pack(v), self.mask)[0]

    def test_matches_native_block_mask(self):
        out = self.run_attention(self.q, self.k, self.v)
        causal, full = [], []
        for start in (0, 5):
            mask = torch.zeros(5, 5, dtype=torch.bool)
            mask[:3, :3] = torch.ones(3, 3, dtype=torch.bool).tril()
            mask[3:, :] = True
            expected = (
                F.scaled_dot_product_attention(
                    self.q[start : start + 5].transpose(0, 1),
                    self.k[start : start + 5].transpose(0, 1),
                    self.v[start : start + 5].transpose(0, 1),
                    attn_mask=mask,
                )
                .transpose(0, 1)
                .flatten(-2)
            )
            causal.append(expected[:3])
            full.append(expected[3:])
        torch.testing.assert_close(out["causal_seq"][:6], torch.cat(causal))
        torch.testing.assert_close(out["full_only_seq"][:4], torch.cat(full))

    def test_future_cannot_affect_ar_or_another_sample(self):
        baseline = self.run_attention(self.q, self.k, self.v)
        k, v = self.k.clone(), self.v.clone()
        k[3:5] += 100
        v[3:5] -= 100
        changed = self.run_attention(self.q, k, v)
        torch.testing.assert_close(changed["causal_seq"], baseline["causal_seq"])
        torch.testing.assert_close(changed["full_only_seq"][2:4], baseline["full_only_seq"][2:4])
        self.assertFalse(torch.allclose(changed["full_only_seq"][:2], baseline["full_only_seq"][:2]))

    def test_fm_loss_alone_updates_ar_weights(self):
        ar_k = torch.nn.Linear(8, 8, bias=False)
        ar_v = torch.nn.Linear(8, 8, bias=False)
        inputs = torch.randn(10, 8)
        k, v = ar_k(inputs).reshape(10, 2, 4), ar_v(inputs).reshape(10, 2, 4)
        # Only AR positions use the trainable projections; FM projections are fixed.
        indices = torch.tensor([0, 1, 2, 5, 6, 7])
        packed_k = self.k.clone().index_copy(0, indices, k[indices])
        packed_v = self.v.clone().index_copy(0, indices, v[indices])
        out = self.run_attention(self.q, packed_k, packed_v)
        F.mse_loss(out["full_only_seq"][:4], torch.ones(4, 8)).backward()
        for module in (ar_k, ar_v):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.norm().item(), 0)


class NativeARTest(unittest.TestCase):
    def test_ce_without_bridge_or_future(self):
        embedding = torch.nn.Embedding(12, 8).to(torch.bfloat16)
        projection = torch.nn.Linear(8, 8).to(torch.bfloat16)
        reasoner = SimpleNamespace(
            embed_tokens=embedding,
            reasoner_forward=lambda **kw: projection(kw["inputs_embeds"]),
        )
        network = SimpleNamespace(
            language_model=SimpleNamespace(
                model=reasoner,
                lm_head=torch.nn.Linear(8, 12).to(torch.bfloat16),
            )
        )
        packed = SimpleNamespace(
            gui_action_target_ids=[[1, 2, 3]],
            gui_action_target_masks=[[True, True, True]],
            gui_multimodal_inputs=[{"input_ids": torch.tensor([[1, 2]])}],
            gui_mot_joint=True,
            gui_native_fm_ar=True,
        )
        prepared = (
            torch.randn(1, 2, 8),
            torch.zeros(1, 2, dtype=torch.bool),
            [],
            torch.arange(2).reshape(1, 1, 2).expand(3, 1, 2),
            None,
        )
        out = {}
        with patch(
            "cosmos_framework.model.generator.reasoner.qwen3_vl.utils.prepare_multimodal_reasoner_inputs",
            return_value=prepared,
        ):
            hybrid_ar_forward(network, packed, torch.zeros(1, 8, dtype=torch.bfloat16), out)
        F.cross_entropy(out["gui_ar_logits"].float().flatten(0, 1), out["gui_ar_labels"].flatten()).backward()
        self.assertGreater(projection.weight.grad.norm().item(), 0)
        self.assertNotIn("gui_ar_base_logits", out)
        self.assertNotIn("gui_ar_max_logit_delta", out)
        self.assertNotIn("gui_ar_residual_norm_ratio", out)

    def make_model(self):
        tokenizer = SimpleNamespace(
            eos_token_id=0,
            convert_tokens_to_ids=lambda _: 0,
            encode=lambda *a, **k: [1],
            decode=lambda *a, **k: '<answer>{"action_type":"Wait","value":"None"}</answer>',
        )
        model = SimpleNamespace(
            native_fm_ar=True,
            video_only=True,
            mot_joint=True,
            hybrid_ar=False,
            config=SimpleNamespace(max_action_dim=64),
            _gui_prefix=[{"input_ids": torch.tensor([[1]])}],
            _gui_processor=lambda: SimpleNamespace(tokenizer=tokenizer),
            _prepare_gui_prefix=Mock(),
            generate_samples_from_batch=Mock(side_effect=AssertionError("FM sampler must not execute")),
            net=Mock(return_value={"ids": [1], "first_max_logit_delta": 0, "finished": True}),
        )
        return model

    def test_policy_never_calls_fm(self):
        model = self.make_model()
        with patch.dict("os.environ", {"GUI_PRECOMPUTE_VISUAL_OUTSIDE_FSDP": "0", "GUI_NATIVE_PREFIX_FILE": ""}):
            output = JointPolicy(model).predict_action(torch.zeros(3, 32, 32, dtype=torch.uint8), "wait")
        model.generate_samples_from_batch.assert_not_called()
        model._prepare_gui_prefix.assert_called_once()
        request = model.net.call_args.kwargs["packed_seq"].gui_ar_decode_request
        self.assertTrue(request["ar_only"])
        self.assertNotIn("future", request)
        self.assertNotIn("plan", request)
        self.assertEqual(output["actions"][0]["action"]["type"], "wait")

    def test_policy_rejects_future_requests(self):
        with self.assertRaisesRegex(ValueError, "AR actions only"):
            JointPolicy(self.make_model()).predict_action(
                torch.zeros(3, 32, 32, dtype=torch.uint8),
                "wait",
                return_future_latents=True,
            )


if __name__ == "__main__":
    unittest.main()
