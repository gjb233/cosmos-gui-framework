"""CPU loss-path regression: explicit scalar objective and gradients."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from cosmos_framework.gui_mot.native_model import GuiJointModel


class NativeLossIntegrationTest(unittest.TestCase):
    def test_native_loss_and_gradients_with_and_without_grouped_ce(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                torch.manual_seed(9)
                logits = torch.randn(1, 4, 6, requires_grad=True)
                vision = torch.randn(1, 2, 2, 2, requires_grad=True)
                labels = torch.tensor([[1, 2, 3, 4]])
                weights = torch.tensor([[0.2, 0.4, 0.4, 0.25]])
                cfg = SimpleNamespace(
                    rectified_flow_training_config=SimpleNamespace(
                        train_time_weight="uniform", independent_action_schedule=False, loss_scale=0.7
                    ),
                    sound_gen=False,
                    lbl=SimpleNamespace(coeff_gen=0, coeff_und=0),
                )
                model = SimpleNamespace(
                    config=cfg,
                    video_only=True,
                    hybrid_ar=False,
                    mot_joint=True,
                    grouped_ce=grouped,
                    hybrid_plan_fm_weight=0.5,
                    hybrid_plan_x0_weight=0.1,
                    hybrid_ce_weight=1.3,
                    hybrid_kd_weight=0.0,
                    _loss_averaging_group=lambda: (None, 1),
                )
                out = {
                    "preds_vision": [vision],
                    "gui_ar_logits": logits,
                    "gui_ar_labels": labels,
                    "gui_ar_token_mask": torch.ones_like(labels, dtype=torch.bool),
                    "gui_ar_token_weights": weights,
                }
                total, metrics = GuiJointModel._compute_losses(
                    model,
                    out,
                    SimpleNamespace(action=None),
                    SimpleNamespace(vt_target_vision=[torch.zeros_like(vision)]),
                    None,
                    False,
                )
                token_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
                ce = (token_loss * weights.flatten()).sum() if grouped else token_loss.mean()
                expected = 0.7 * vision[:, 1:].square().mean() + 1.3 * ce
                torch.testing.assert_close(total, expected)
                actual_grad = torch.autograd.grad(total, (logits, vision), retain_graph=True)
                expected_grad = torch.autograd.grad(expected, (logits, vision))
                for actual, expected in zip(actual_grad, expected_grad, strict=True):
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(float(actual_grad[1][:, :1].abs().sum()), 0.0)
                self.assertEqual(
                    set(metrics), {"gui_ar_ce", "flow_matching_loss_vision", "flow_matching_loss_vision_per_instance"}
                )


if __name__ == "__main__":
    unittest.main()
