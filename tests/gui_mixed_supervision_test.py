import json
import re
import unittest

import torch

from cosmos_framework.gui_mot.mixed_supervision import (
    answer_fields,
    canonical_response,
    direct_query,
    direct_response,
    grouped_ce,
    prepare_grouped_batch,
    tokenize_grouped_targets,
)
from cosmos_framework.gui_mot.official_format import (
    RESPONSE_TEMPLATE,
    SYSTEM_PROMPT,
    official_action_text,
    official_query,
    supervised_query,
    supervised_system_prompt,
)


class CharTokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(x) for x in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class MixedSupervisionTest(unittest.TestCase):
    def test_direct_has_no_empty_reasoning_and_uses_gt_coordinates(self):
        text = direct_response({"type": "click", "x": 0.5, "y": 1.0})
        self.assertNotIn("thinking", text)
        self.assertNotIn("thinking", direct_query(RESPONSE_TEMPLATE))
        body = json.loads(text.split("<answer>")[1].split("</answer>")[0])
        self.assertEqual(body["point_2d"], [500, 1000])
        self.assertEqual(body["action_target"], "None")

    def test_fixed_fields_and_eos_receive_supervision(self):
        text = direct_response({"type": "input_text", "text": 'a "quote" and {brace}'})
        spans = answer_fields(text)
        ids, weights = tokenize_grouped_targets(CharTokenizer(), [text], return_masks=True)
        self.assertEqual(ids[0][-1], 0)
        self.assertTrue(all(weight > 0 for weight in weights[0]))
        self.assertAlmostEqual(sum(weights[0]), 1.25)
        a, b = spans["value"]
        self.assertEqual(json.loads(text[a:b]), 'a "quote" and {brace}')
        self.assertAlmostEqual(sum(weights[0][a + 1 : b - 1]), 0.5)

    def test_long_reasoning_does_not_dilute_semantic_weights(self):
        answer = direct_response({"type": "wait"})
        for count in (2, 400):
            text = "<thinking>" + "x" * count + "</thinking>\n" + answer
            _, weights = tokenize_grouped_targets(CharTokenizer(), [text], return_masks=True)
            a, b = answer_fields(text)["action_type"]
            self.assertAlmostEqual(sum(weights[0][a + 1 : b - 1]), 1.0)
            self.assertAlmostEqual(sum(weights[0]), 1.5)

    def test_per_axis_mass_is_independent_of_digits_and_json_layout(self):
        for kind in ("Click", "LongPress"):
            for point in ([5, 7], [500, 250], [0, 1000]):
                for indent in (None, 2, 4):
                    body = {"action_type": kind, "action_target": "None", "value": "None", "point_2d": point}
                    text = "<answer>\n" + json.dumps(body, indent=indent) + "\n</answer>"
                    _, weights = tokenize_grouped_targets(CharTokenizer(), [text], return_masks=True)
                    spans = answer_fields(text)
                    a, b = spans["point_2d"]
                    numbers = list(re.finditer(r"\d+", text[a:b]))
                    semantic = set()
                    for number in numbers:
                        left, right = a + number.start(), a + number.end()
                        self.assertAlmostEqual(sum(weights[0][left:right]), 0.4)
                        semantic.update(range(left, right))
                    a, b = spans["action_type"]
                    self.assertAlmostEqual(sum(weights[0][a + 1 : b - 1]), 0.2)
                    semantic.update(range(a + 1, b - 1))
                    self.assertAlmostEqual(sum(w for i, w in enumerate(weights[0]) if i not in semantic), 0.25)

    def test_canonicalization_preserves_actions_and_reasoning(self):
        text = "<thinking>The blue button is the target.</thinking>\n" + direct_response(
            {"type": "click", "x": 0.123, "y": 0.456}
        ).replace('"action_target": "None"', '"action_target": "blue button"')
        fixed = canonical_response(text)
        self.assertIn("The blue button is the target.", fixed)
        self.assertIn('"action_target": "None"', fixed)
        before, after = answer_fields(text), answer_fields(fixed)
        for key in ("action_type", "value", "point_2d"):
            self.assertEqual(text[slice(*before[key])], fixed[slice(*after[key])])
        ids = tokenize_grouped_targets(CharTokenizer(), [text])[0]
        self.assertEqual("".join(chr(i) for i in ids[:-1]), fixed)

    def test_training_and_inference_use_identical_fixed_prompts(self):
        query = official_query({"step_instruction": "Tap blue", "previous_actions": []}, (1080, 2400))
        action = official_action_text({"type": "click", "x": 0.5, "y": 0.5}, "Tap blue")
        source = {"ai_caption": [query], "gui_system_prompt": [[SYSTEM_PROMPT]], "gui_action_text": [[action]]}
        train = prepare_grouped_batch(source)
        infer = prepare_grouped_batch({**source, "gui_action_text": [[""]]})
        self.assertEqual(train["ai_caption"], infer["ai_caption"])
        self.assertEqual(train["gui_system_prompt"], infer["gui_system_prompt"])
        self.assertEqual(train["ai_caption"][0], supervised_query(query))
        self.assertEqual(train["gui_system_prompt"][0], supervised_system_prompt())
        self.assertNotIn('"the description of the target', train["ai_caption"][0])
        self.assertNotIn("action_target: Element description", train["gui_system_prompt"][0])
        self.assertEqual(prepare_grouped_batch(train), train)
        self.assertEqual(source["gui_action_text"], [[action]])
        self.assertNotIn("<thinking>", train["gui_action_text"][0])

    def test_reasoning_prompt_matches_nonempty_reasoning_target(self):
        response = "<thinking>Tap the blue button.</thinking>\n" + direct_response({"type": "wait"})
        batch = {"ai_caption": [RESPONSE_TEMPLATE], "gui_action_text": [response]}
        train = prepare_grouped_batch(batch)
        infer = prepare_grouped_batch({**batch, "gui_action_text": [""]}, response_style="reasoning")
        self.assertEqual(train["ai_caption"], infer["ai_caption"])
        self.assertIn("<thinking>", train["ai_caption"][0])

    def test_invalid_coordinates_fail_instead_of_silently_losing_supervision(self):
        for point in ([1001, 0], [-1, 4], [True, 2], [1.2, 4], None):
            body = {"action_type": "Click", "action_target": "None", "value": "None", "point_2d": point}
            with self.assertRaises(ValueError):
                tokenize_grouped_targets(CharTokenizer(), ["<answer>" + json.dumps(body) + "</answer>"])

    def test_literal_tags_and_empty_write_values_are_supervised_as_data(self):
        for value in ("", "<thinking>literal</thinking>", "literal </answer> text", '<answer>{"x":1}</answer>'):
            text = direct_response({"type": "input_text", "text": value})
            batch = prepare_grouped_batch({"gui_action_text": [text], "ai_caption": [RESPONSE_TEMPLATE]})
            self.assertNotIn("<thinking>", batch["ai_caption"][0])
            a, b = answer_fields(text)["value"]
            self.assertEqual(json.loads(text[a:b]), value)
            _, weights = tokenize_grouped_targets(CharTokenizer(), [text], return_masks=True)
            self.assertAlmostEqual(sum(weights[0]), 1.25)
            # Empty JSON string has no content characters; its quotes encode
            # the meaningful empty value and must receive semantic supervision.
            semantic = weights[0][a + 1 : b - 1] if value else weights[0][a:b]
            self.assertAlmostEqual(sum(semantic), 0.5)

    def test_duplicate_last_field_and_trailing_payload_are_rejected(self):
        text = direct_response({"type": "wait"})
        duplicate = text.replace('"point_2d": null', '"point_2d": null, "point_2d": [1, 2]')
        for malformed in (duplicate, text + "extra", text.replace("</answer>", "")):
            with self.assertRaises(ValueError):
                answer_fields(malformed)

    def test_masked_labels_have_no_gradient(self):
        logits = torch.randn(1, 4, 6, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, 4]])
        weights = torch.tensor([[0.5, 0.0, 0.5, 0.25]])
        grouped_ce(logits, labels, weights).backward()
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 3].abs().sum()), 0)

    def test_long_target_fails_instead_of_truncating_action(self):
        text = "<thinking>" + "x" * 1100 + "</thinking>" + direct_response({"type": "wait"})
        with self.assertRaises(ValueError):
            tokenize_grouped_targets(CharTokenizer(), [text], return_masks=True)


if __name__ == "__main__":
    unittest.main()
