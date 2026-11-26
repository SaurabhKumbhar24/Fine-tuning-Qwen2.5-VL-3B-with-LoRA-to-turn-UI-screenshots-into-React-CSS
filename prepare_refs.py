"""
Phase 3, step 1: save the Flame-Eval-React reference material.

Writes:
    outputs/eval/refs/<problem_id>.png   the 80 input screenshots
    outputs/eval/gold.jsonl              the human-written answers, in the same
                                         format as evaluate.py's output

Run once:  python prepare_refs.py
"""
import os
import json

from datasets import load_dataset

EVAL_DATASET = "Flame-Code-VLM/Flame-Eval-React"
EVAL_DIR = "outputs/eval/"
REFS_DIR = os.path.join(EVAL_DIR, "refs")


def main():
    os.makedirs(REFS_DIR, exist_ok=True)
    ds = load_dataset(EVAL_DATASET, split="train")

    with open(os.path.join(EVAL_DIR, "gold.jsonl"), "w", encoding="utf-8") as f:
        for row in ds:
            row["image"].convert("RGB").save(os.path.join(REFS_DIR, f"{row['problem_id']}.png"))

            record = {
                "problem_id": row["problem_id"],
                "complexity_level": row["complexity_level"],
                "num_tokens": None,
                "hit_max_tokens": False,
                "jsx": row["component"].strip(),
                "css": row["style"].strip(),
                "raw": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(ds)} screenshots to {REFS_DIR} and answers to {EVAL_DIR}gold.jsonl")


if __name__ == "__main__":
    main()
