"""GT-preserving direct/reasoning supervision for native GUI SFT."""

import hashlib
import json
import random
import re
from pathlib import Path

from .joint_dataset import JointPolicyDataset, StreamingJointPolicyDataset
from .official_format import direct_query, official_action_text, supervised_query, supervised_system_prompt


def direct_response(action):
    return canonical_response(official_action_text(action, ""))


def answer_fields(text):
    """Return JSON value spans without interpreting braces inside strings."""
    # Parse the JSON before looking for the closing envelope: a Write value
    # may itself contain literal </answer>, braces or other control-looking text.
    opening = re.match(r"\s*(?:<thinking>.*?</thinking>\s*)?<answer>\s*", text, re.S)
    if opening is None:
        raise ValueError("Missing answer envelope")
    start = opening.end()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate action field")
            value[key] = item
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    parsed, end = decoder.raw_decode(text, start)
    if text[end:].strip() not in ("</answer>", "</answer><|im_end|>"):
        raise ValueError("Invalid closing answer envelope")
    if not isinstance(parsed, dict) or set(parsed) != {"action_type", "action_target", "value", "point_2d"}:
        raise ValueError("Expected the four official action fields")
    cursor = start
    while text[cursor].isspace():
        cursor += 1
    cursor += 1  # opening object brace
    spans = {}
    for _ in parsed:
        while text[cursor].isspace() or text[cursor] == ",":
            cursor += 1
        key, cursor = decoder.raw_decode(text, cursor)
        while text[cursor].isspace():
            cursor += 1
        if text[cursor] != ":":
            raise ValueError("Invalid JSON field separator")
        cursor += 1
        while text[cursor].isspace():
            cursor += 1
        value_start = cursor
        _, cursor = decoder.raw_decode(text, cursor)
        if key in spans:
            raise ValueError("Duplicate action field")
        spans[key] = (value_start, cursor)
    return spans


def canonical_response(text):
    """Preserve actions and nonempty reasoning; fix the unsupervised prefix."""
    spans = answer_fields(text)
    start, end = spans["action_target"]
    text = text[:start] + '"None"' + text[end:]
    return re.sub(r"^\s*<thinking>\s*</thinking>\s*", "", text, count=1)


def prepare_grouped_batch(batch, *, response_style="direct"):
    """Align target, user prompt and system prompt before prefix tokenization."""
    if response_style not in ("direct", "reasoning"):
        raise ValueError("Grouped inference response style must be direct or reasoning")

    def flatten(value):
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return value

    texts = flatten(batch["gui_action_text"])
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Expected action text strings")
    if any(texts) and not all(texts):
        raise ValueError("Cannot mix training and inference action texts")
    texts = [canonical_response(text) if text else "" for text in texts]
    captions = flatten(batch["ai_caption"])
    systems = flatten(batch.get("gui_system_prompt", [None] * len(texts)))
    result = dict(batch)
    result["gui_action_text"] = texts
    result["ai_caption"] = [
        supervised_query(
            caption, reasoning=(bool(re.match(r"\s*<thinking>", text)) if text else response_style == "reasoning")
        )
        for caption, text in zip(captions, texts, strict=True)
    ]
    result["gui_system_prompt"] = [supervised_system_prompt(system) for system in systems]
    if len(systems) != len(texts):
        raise ValueError("System prompt and action counts differ")
    return result


def tokenize_grouped_targets(tokenizer, texts, *, return_masks=False):
    """Semantic CE with fixed per-axis mass, plus .25 format and .25 reasoning.

    Click/LongPress: type=.2, x=.4, y=.4. Other actions: type/value=.5/.5
    when a value is applicable, otherwise type=1. JSON syntax and fixed
    placeholders have nonzero format supervision. Total mass remains 1.25
    for direct responses and 1.5 for responses with nonempty reasoning.
    """
    all_ids, all_weights = [], []
    for original in texts:
        text = canonical_response(original)
        spans = answer_fields(text)
        action_kind = json.loads(text[slice(*spans["action_type"])])

        def content_span(key):
            start, end = spans[key]
            if text[start] == '"' and end - start > 2:
                start, end = start + 1, end - 1
            return start, end

        semantic = {"type": content_span("action_type")}
        scales = {"type": 1.0, "thinking": 0.25, "format": 0.25}
        if action_kind in ("Click", "LongPress"):
            start, end = spans["point_2d"]
            match = re.fullmatch(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text[start:end])
            if match is None or any(not 0 <= int(match[i]) <= 1000 for i in (1, 2)):
                raise ValueError("Click coordinates must be two integers in [0, 1000]")
            semantic.update({axis: (start + match.start(i), start + match.end(i)) for axis, i in (("x", 1), ("y", 2))})
            scales.update(type=0.2, x=0.4, y=0.4)
        elif action_kind in ("Write", "Scroll", "OpenApp", "Terminate"):
            semantic["value"] = content_span("value")
            scales.update(type=0.5, value=0.5)
        thought = re.match(r"\s*<thinking>(.*?)</thinking>", text, re.S)
        thinking_span = thought.span(1) if thought else None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Native SFT requires an EOS token")
        if not ids or ids[-1] != eos:
            ids.append(eos)
            offsets.append((len(text), len(text)))
        if len(ids) > 1024:
            raise ValueError("Response exceeds the validated 1024-token budget; never truncate the action")
        groups = []
        for index, (left, right) in enumerate(offsets):

            def overlaps(span):
                return span is not None and right > span[0] and left < span[1]

            selected = [group for group, span in semantic.items() if overlaps(span)]
            if len(selected) > 1:
                raise ValueError("Tokenizer merges separate semantic fields into one token")
            if index == len(ids) - 1 and ids[index] == eos:
                groups.append("format")
            elif selected:
                groups.append(selected[0])
            elif overlaps(thinking_span):
                groups.append("thinking")
            else:
                groups.append("format")
        counts = {group: groups.count(group) for group in scales}
        if any(not counts[group] for group in (*semantic, "format")):
            raise ValueError("Missing semantic or format supervision")
        weights = [scales[group] / counts[group] for group in groups]
        all_ids.append(ids)
        all_weights.append(weights)
    return (all_ids, all_weights) if return_masks else all_ids


def grouped_ce(logits, labels, weights):
    import torch.nn.functional as F

    if weights.shape != labels.shape or logits.shape[:2] != labels.shape:
        raise ValueError("Grouped CE shapes differ")
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    return (losses * weights).sum(dim=-1).mean()


class MixedJointPolicyDataset(JointPolicyDataset):
    def __init__(self, manifest, *, supervision_cache, shuffle_seed=42, **kwargs):
        super().__init__(manifest, **kwargs)
        cache = json.loads(Path(supervision_cache).read_text())
        actual = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        if actual != cache["manifest_sha256"]:
            raise ValueError("Supervision cache does not match the training manifest")
        self.reasoning = cache["responses"]
        self.entries = [(i, False) for i in range(len(self.windows))]
        self.entries += [
            (i, True)
            for i, window in enumerate(self.windows)
            if f"{window[0].episode_id}:{window[0].step}" in self.reasoning
        ]
        random.Random(shuffle_seed).shuffle(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        source_index, reasoning = self.entries[index]
        sample = super().__getitem__(source_index)
        row = self.windows[source_index][0]
        if reasoning:
            sample["gui_action_text"] = canonical_response(self.reasoning[f"{row.episode_id}:{row.step}"])
        else:
            sample["gui_action_text"] = direct_response(row.action)
            sample["ai_caption"] = direct_query(sample["ai_caption"])
        return sample


class MixedStreamingPolicyDataset(StreamingJointPolicyDataset):
    def __init__(self, *args, **kwargs):
        # Preserve the parent's deterministic rank/resume iterator.
        self.dataset = MixedJointPolicyDataset(*args, **kwargs)
        self.shard_rank = 0
        self.shard_world_size = 1
        self.start_sample = 0
