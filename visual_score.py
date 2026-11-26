"""
Phase 3, step 3: compare each rendered page with the input screenshot.

Scores per row:
    CLIP  - do the two pictures look like the same page? (cosine similarity of
            CLIP image embeddings, about 0.5-1.0)
    SSIM  - is the layout the same? (structural similarity of the grey pixels, 0-1)
A page that did not render scores 0 on both.

Run after render.mjs, with one or more run names:
    python visual_score.py gold base finetuned

Writes:
    outputs/eval/<name>.visual.jsonl          scores per row
    outputs/eval/compare/<name>/<id>.png      input | render, side by side
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from transformers import CLIPModel, CLIPProcessor

EVAL_DIR = "outputs/eval/"
REFS_DIR = os.path.join(EVAL_DIR, "refs")
CLIP_ID = "openai/clip-vit-base-patch32"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def clip_embed(model, processor, images):
    inputs = processor(images=images, return_tensors="pt")
    features = model.get_image_features(**inputs)
    # transformers 5 returns an output object; older versions return the tensor
    features = getattr(features, "pooler_output", features)
    return features / features.norm(dim=-1, keepdim=True)   # length 1 -> dot product = cosine


def ssim(ref, render):
    a = np.asarray(ref.convert("L"))
    b = np.asarray(render.convert("L").resize(ref.size))
    return float(structural_similarity(a, b, data_range=255))


def side_by_side(ref, render, path):
    width, height = ref.size
    canvas = Image.new("RGB", (width * 2 + 10, height), "white")
    canvas.paste(ref, (0, 0))
    if render is not None:
        canvas.paste(render.resize(ref.size), (width + 10, 0))
    else:
        ImageDraw.Draw(canvas).text((width + 30, 30), "DID NOT RENDER", fill="red")
    canvas.save(path)


def score_run(name, model, processor, ref_embeddings):
    rows = read_jsonl(os.path.join(EVAL_DIR, f"{name}.render.jsonl"))
    compare_dir = os.path.join(EVAL_DIR, "compare", name)
    os.makedirs(compare_dir, exist_ok=True)

    results = []
    for row in rows:
        pid = row["problem_id"]
        ref = Image.open(os.path.join(REFS_DIR, f"{pid}.png")).convert("RGB")
        render = None
        clip_score, ssim_score = 0.0, 0.0

        if row["render_ok"]:
            render = Image.open(os.path.join(EVAL_DIR, "renders", name, f"{pid}.png")).convert("RGB")
            render_embedding = clip_embed(model, processor, [render])[0]
            clip_score = float(render_embedding @ ref_embeddings[pid])
            ssim_score = ssim(ref, render)

        side_by_side(ref, render, os.path.join(compare_dir, f"{pid}.png"))
        results.append({
            "problem_id": pid,
            "complexity_level": row["complexity_level"],
            "render_ok": row["render_ok"],
            "clip": round(clip_score, 4),
            "ssim": round(ssim_score, 4),
        })

    with open(os.path.join(EVAL_DIR, f"{name}.visual.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def summary_line(label, rows):
    n = len(rows)
    rendered = [r for r in rows if r["render_ok"]]
    mean = lambda key, rs: sum(r[key] for r in rs) / len(rs) if rs else 0.0
    return (f"{label:<22} {len(rendered):>3}/{n:<3} {mean('clip', rows):>7.3f} {mean('ssim', rows):>7.3f}"
            f"   {mean('clip', rendered):>8.3f} {mean('ssim', rendered):>8.3f}")


def main():
    names = sys.argv[1:]
    if not names:
        print("Usage: python visual_score.py gold base finetuned")
        sys.exit(1)

    model = CLIPModel.from_pretrained(CLIP_ID).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    # embed the 80 input screenshots once, reuse them for every run
    ref_ids = sorted(f[:-4] for f in os.listdir(REFS_DIR) if f.endswith(".png"))
    ref_images = [Image.open(os.path.join(REFS_DIR, f"{pid}.png")).convert("RGB") for pid in ref_ids]
    ref_embeddings = dict(zip(ref_ids, clip_embed(model, processor, ref_images)))

    all_results = {name: score_run(name, model, processor, ref_embeddings) for name in names}

    print(f"\n{'':<22} {'render':>7} {'CLIP':>7} {'SSIM':>7}   {'CLIP*':>8} {'SSIM*':>8}")
    for name, rows in all_results.items():
        print(summary_line(name, rows))
        for level in sorted({r["complexity_level"] for r in rows}):
            print(summary_line(f"  complexity {level}", [r for r in rows if r["complexity_level"] == level]))
    print("\nCLIP / SSIM: failed renders count as 0.   CLIP* / SSIM*: rendered pages only.")
    print(f"Side-by-side pictures in {EVAL_DIR}compare/<name>/")


if __name__ == "__main__":
    main()
