"""
Phase 1 of evaluation: generate React + CSS for every Flame-Eval-React screenshot
and save the answers to a .jsonl file (one JSON object per line).

Run it twice and compare:
    python evaluate.py --name base
    python evaluate.py --name finetuned --adapter outputs/qwen2.5vl-lora/final/

Quick test on the first 3 rows:
    python evaluate.py --name test --adapter outputs/qwen2.5vl-lora/final/ --limit 3
"""
import os
import re
import json
import time
import argparse

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from prepare import SYSTEM_PROMPT, USER_PROMPT

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
EVAL_DATASET = "Flame-Code-VLM/Flame-Eval-React"
MAX_PIXELS = 1280 * 28 * 28      # same as training
MAX_NEW_TOKENS = 4096            # longest answers in training were about this long
EVAL_DIR = "outputs/eval/"

JSX_PATTERN = re.compile(r"```jsx\s*\n(.*?)```", re.DOTALL)
CSS_PATTERN = re.compile(r"```css\s*\n(.*?)```", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="name of this run, used for the output file")
    parser.add_argument("--adapter", default=None, help="LoRA folder; leave out to test the base model")
    parser.add_argument("--base", default=MODEL_ID, help="base model id or folder")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


def load_model(base_id, adapter_dir, device):
    model = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()   # fold LoRA into the weights: faster generation
    model.to(device)
    model.eval()
    return model


def build_prompt(processor, image):
    # Same messages as in training, but WITHOUT the assistant answer.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, processor, image, device, max_new_tokens):
    text = build_prompt(processor, image)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]                 # drop the prompt, keep the answer
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return answer, new_tokens.shape[1]


def extract_code(answer):
    jsx = JSX_PATTERN.search(answer)
    css = CSS_PATTERN.search(answer)
    return (jsx.group(1).strip() if jsx else None,
            css.group(1).strip() if css else None)


def already_done(out_file):
    # Lets a crashed run continue where it stopped.
    if not os.path.exists(out_file):
        return set()
    with open(out_file, encoding="utf-8") as f:
        return {json.loads(line)["problem_id"] for line in f if line.strip()}


def summarize(out_file):
    with open(out_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    both = sum(r["jsx"] is not None and r["css"] is not None for r in records)
    cut_off = sum(r["hit_max_tokens"] for r in records)

    print(f"\nRows:               {total}")
    print(f"Both blocks found:  {both} ({both / total:.0%})")
    print(f"Hit max_new_tokens: {cut_off}")
    for level in sorted({r["complexity_level"] for r in records}):
        rows = [r for r in records if r["complexity_level"] == level]
        ok = sum(r["jsx"] is not None and r["css"] is not None for r in rows)
        print(f"  complexity {level}: {ok}/{len(rows)} correct format")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_file = os.path.join(EVAL_DIR, f"{args.name}.jsonl")

    processor = AutoProcessor.from_pretrained(args.base, max_pixels=MAX_PIXELS)
    model = load_model(args.base, args.adapter, device)

    ds = load_dataset(EVAL_DATASET, split="train")
    if args.limit is not None:
        ds = ds.select(range(args.limit))

    done = already_done(out_file)
    print(f"{len(ds)} rows, {len(done)} already done, writing to {out_file}")

    with open(out_file, "a", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if row["problem_id"] in done:
                continue

            start = time.time()
            image = row["image"].convert("RGB")
            answer, num_tokens = generate(model, processor, image, device, args.max_new_tokens)
            jsx, css = extract_code(answer)

            record = {
                "problem_id": row["problem_id"],
                "complexity_level": row["complexity_level"],
                "num_tokens": num_tokens,
                "hit_max_tokens": num_tokens >= args.max_new_tokens,
                "jsx": jsx,
                "css": css,
                "raw": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()   # save to disk now, in case the run crashes

            status = "ok" if jsx and css else "BAD FORMAT"
            print(f"[{i + 1}/{len(ds)}] {row['problem_id']}  {num_tokens} tokens  "
                  f"{time.time() - start:.0f}s  {status}")

    summarize(out_file)


if __name__ == "__main__":
    main()
