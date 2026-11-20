import os
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.gui_mot.joint_dataset import StreamingJointPolicyDataset
from cosmos_framework.gui_mot.resumable_loader import GuiPackingDataLoader


def stream(rank=1, world=3):
    ds = StreamingJointPolicyDataset.__new__(StreamingJointPolicyDataset)
    ds.dataset = list(range(17))
    ds.shard_rank = rank
    ds.shard_world_size = world
    ds.start_sample = 0
    return ds


class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_across_wraparound(self):
        for rank in range(3):
            baseline = iter(stream(rank))
            expected = [next(baseline) for _ in range(30)]
            ds = stream(rank)
            ds.set_start_sample(11)
            resumed = iter(ds)
            self.assertEqual([next(resumed) for _ in range(19)], expected[11:])

    def test_packer_discards_prewarm_and_restores_inner_position(self):
        ds = stream()
        inner = SimpleNamespace(dataset=ds, dataloader=SimpleNamespace(num_workers=0))

        class Inner:
            dataset = ds
            dataloader = inner.dataloader

            def __iter__(self):
                return iter(self.dataset)

        loader = GuiPackingDataLoader.__new__(GuiPackingDataLoader)
        loader.max_samples_per_batch = 1
        loader.dataloader_list = [Inner()]
        loader.dataloaders = [iter(ds)]
        loader.buffers = [deque([next(loader.dataloaders[0])])]
        loader.set_start_iteration(4)
        self.assertEqual(loader.global_id, 4)
        self.assertFalse(loader.buffers[0])
        self.assertEqual(next(loader.dataloaders[0]), 13)


class CheckpointTests(unittest.TestCase):
    def test_reasoner_recompute_matches_outputs_and_gradients(self):
        from cosmos_framework.model.generator.mot.unified_mot import _impl_reasoner_forward

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def reasoner_forward(self, h, cos, sin, cache, idx):
                return h + torch.nn.functional.gelu(self.linear(h))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
                self.norm = torch.nn.LayerNorm(8)

            def rotary_emb(self, h, position_ids):
                return torch.ones_like(h), torch.zeros_like(h)

        torch.manual_seed(7)
        model = Model().train()
        x = torch.randn(1, 5, 8)
        position = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5)
        values = []
        for enabled in ("0", "1"):
            model.zero_grad(set_to_none=True)
            with patch.dict(os.environ, {"GUI_AR_ACTIVATION_CHECKPOINT": enabled}):
                y = _impl_reasoner_forward(model, None, None, position_ids=position, inputs_embeds=x)
                y.square().sum().backward()
            values.append((y.detach(), [p.grad.clone() for p in model.parameters()]))
        torch.testing.assert_close(values[0][0], values[1][0])
        for a, b in zip(values[0][1], values[1][1]):
            torch.testing.assert_close(a, b)


class AnswerMaskTests(unittest.TestCase):
    def test_answer_mask_is_independent_of_newlines(self):
        from cosmos_framework.gui_mot.hybrid_action import tokenize_action_targets

        class Tokenizer:
            eos_token_id = 0

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        for spacing in ("", "\n", " \n  "):
            text = "<thinking>example</thinking><answer>" + spacing + '{"action_type":"Wait"}' + spacing + "</answer>"
            ids, masks = tokenize_action_targets(Tokenizer(), [text], return_masks=True)
            supervised = "".join(chr(i) for i, selected in zip(ids[0], masks[0]) if selected)
            self.assertEqual(supervised, '{"action_type":"Wait"}')
            self.assertFalse(masks[0][-1])
        with self.assertRaises(ValueError):
            tokenize_action_targets(Tokenizer(), ['<answer>{"action_type":"Wait"}'], return_masks=True)


class CoordinateTests(unittest.TestCase):
    def test_official_coordinates_round_trip_and_accept_boundary(self):
        from cosmos_framework.gui_mot.joint_policy import parse_native_tool_calls
        from cosmos_framework.gui_mot.official_format import official_action_text, official_history_action

        for kind in ("click", "long_press"):
            for x, y in ((0.5, 0.25), (1.0, 0.0)):
                action = {"type": kind, "x": x, "y": y}
                text = official_action_text(action, "test")
                parsed = parse_native_tool_calls(text)[0]["action"]
                self.assertEqual(parsed, action)
        self.assertIn("[500, 250]", official_history_action({"type": "click", "x": 0.5, "y": 0.25}))
        self.assertFalse(parse_native_tool_calls('<answer>{"action_type":"Click","point_2d":[1001,500]}</answer>'))


if __name__ == "__main__":
    unittest.main()
