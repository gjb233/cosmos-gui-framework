"""Run GUI-Libra's official AndroidControl-v2 398-step protocol locally."""

import argparse
import json
import re
import time
from pathlib import Path

from .official_format import SYSTEM_PROMPT, official_query


def extract_plan_fields(text):
    def field(name, default=""):
        match = re.search(rf'"{name}"\s*:\s*"([^"]*)"', text, re.S)
        return match.group(1).strip() if match else default

    point_match = re.search(r'"point_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', text)
    point = [int(point_match.group(1)), int(point_match.group(2))] if point_match else None
    return {
        "action_type": field("action_type"),
        "element_description": field("action_target"),
        "value": field("value"),
        "point_2d": point,
    }


def evaluate(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    official = json.loads(Path(args.official_samples).read_text(encoding="utf-8"))
    selected = official[args.shard_index :: args.num_shards]
    if args.max_samples:
        selected = selected[: args.max_samples]
    screenshot_dir = Path(args.screenshot_dir)
    image_paths = [screenshot_dir / row["screenshot"] for row in selected]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official screenshots are missing: {missing[:10]}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .eval()
        .to("cuda")
    )

    results = []
    started = time.time()
    with (output / "predictions.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for offset in range(0, len(selected), args.batch_size):
            batch = selected[offset : offset + args.batch_size]
            prompts, images = [], []
            for row in batch:
                # Use the exact benchmark screenshot referenced by the official
                # sample. Video frames have different geometry and invalidate
                # accessibility-tree bbox scoring even with relative coords.
                with Image.open(screenshot_dir / row["screenshot"]) as source:
                    image = source.convert("RGB")
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": official_query(row, image.size)},
                        ],
                    },
                ]
                prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                images.append(image)
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            for row, token_ids in zip(batch, generated[:, prompt_length:]):
                ids = token_ids.tolist()
                response = processor.tokenizer.decode(ids, skip_special_tokens=False)
                parsed = extract_plan_fields(response)
                result = {
                    "episode_id": row["episode_id"],
                    "step": row["step"],
                    "instruction": row["step_instruction"],
                    "action": "",
                    **parsed,
                    "reason": "",
                    "response": response,
                    "finish_reason": "stop" if processor.tokenizer.eos_token_id in ids else "length",
                    "output_tokens": len(ids),
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
            summary = {
                "expected": len(selected),
                "evaluated": len(results),
                "parse_errors": sum(not row["action_type"] for row in results),
                "truncated": sum(row["finish_reason"] == "length" for row in results),
                "elapsed_seconds": time.time() - started,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-samples", required=True)
    parser.add_argument(
        "--screenshot-dir",
        required=True,
        help="Root of the extracted official AndroidControl_images archive",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
