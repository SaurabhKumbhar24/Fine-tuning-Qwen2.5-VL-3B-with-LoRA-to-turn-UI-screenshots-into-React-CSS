# Screenshot to React

Fine-tuning **Qwen2.5-VL-3B-Instruct** with **LoRA** to turn a UI screenshot into a React component plus its CSS.

The training loop is written by hand in plain PyTorch, with no `Trainer` or TRL, to show every step of fine-tuning a vision-language model. The evaluation renders the generated pages in a headless browser and compares them with the input screenshots.

```
screenshot.png  ──►  Qwen2.5-VL-3B + LoRA  ──►  ```jsx ... ```  +  ```css ... ```
```

## Results

*Coming after the training run.* The table will compare the base model, the fine-tuned model and the human-written reference answers on Flame-Eval-React:

| | Format | JSX parses | Renders | CLIP | SSIM |
|---|---|---|---|---|---|
| Base Qwen2.5-VL-3B | | | | | |
| Fine-tuned (LoRA) | | | | | |
| Reference answers | 100% | 99% | 79/80 | 0.957 | 0.939 |

The reference answers set the upper limit. One of the 80 reference answers has a syntax error in the dataset itself, so no model can be compared on that row against a working reference.

## Data

| | Dataset | Rows | Use |
|---|---|---|---|
| Training | [Flame-Code-VLM/Flame-Waterfall-React](https://huggingface.co/datasets/Flame-Code-VLM/Flame-Waterfall-React), single-image file | 166,000 | 95% train, 5% validation |
| Test | [Flame-Code-VLM/Flame-Eval-React](https://huggingface.co/datasets/Flame-Code-VLM/Flame-Eval-React) | 80 | never trained on |

Each training example is a chat: a system prompt, the screenshot with a short instruction, and the answer as two code blocks (`jsx`, then `css`). Examples longer than 4,096 tokens are dropped rather than cut off. That removes about 1.8% of the data.

## Training setup

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct`, bf16, SDPA attention |
| Image size | at most 1280 × 28 × 28 pixels (≤ 1,280 image tokens) |
| LoRA | r = 16, alpha = 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` (7.4 M trainable parameters, 0.2%) |
| Loss | only on the answer tokens; the prompt, image and padding are masked with -100 |
| Batch | 2 × 8 gradient accumulation = 16 |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01, gradient clipping 1.0 |
| Schedule | 5% linear warmup, then cosine decay |
| Memory | gradient checkpointing, bf16 autocast |
| Validation | every 100 optimizer steps; the adapter with the lowest validation loss is saved |

## Evaluation

Three phases, each with its own script:

1. **Generate** (`evaluate.py`, GPU). Greedy decoding on all 80 test screenshots. It checks whether the answer has exactly the `jsx` and `css` blocks.
2. **Syntax** (`syntax_check.mjs`). Does the JSX parse (`@babel/parser`)? Does it `export default` a component? Does the CSS parse (`postcss`)? Which packages does it import?
3. **Visual** (`render.mjs` and `visual_score.py`):
   - Each answer is bundled with esbuild and opened in headless Chromium at the screenshot's size, then screenshotted.
   - The screenshot is compared with the input image using **CLIP** image similarity (does it look like the same page?) and **SSIM** (is the layout the same?).
   - Pages that don't render score 0.
   - For scale: two *different* pages score about CLIP 0.70 and SSIM 0.78.

## Project layout

| File | What it does |
|---|---|
| `prepare.py` | Downloads Flame-Waterfall-React, splits it and builds the chat-format dataset (`data/flame_clean`) |
| `train.py` | Length filter (`data/filtered_flame`), collator, LoRA, training and validation loop, saving checkpoints |
| `evaluate.py` | Phase 1: generates answers for Flame-Eval-React and writes `outputs/eval/<name>.jsonl` |
| `syntax_check.mjs` | Phase 2: syntax checks |
| `prepare_refs.py` | Saves the test screenshots and the reference answers (`gold.jsonl`) |
| `render.mjs` | Phase 3: builds and screenshots every answer |
| `visual_score.py` | Phase 3: CLIP and SSIM scores, plus side-by-side images |

## How to run

**Training and generation (GPU, tested setup: one 24 GB GPU)**
```bash
pip install -r requirements.txt          # plus a CUDA build of torch and torchvision
python train.py                          # first run also builds the datasets (prepare.py)
python evaluate.py --name base
python evaluate.py --name finetuned --adapter outputs/qwen2.5vl-lora/checkpoint_<N>/
```

**Syntax and visual checks (CPU, Node.js 18+)**
```bash
npm install
npx playwright install chromium
python prepare_refs.py
node syntax_check.mjs outputs/eval/finetuned.jsonl
node render.mjs outputs/eval/gold.jsonl
node render.mjs outputs/eval/base.jsonl
node render.mjs outputs/eval/finetuned.jsonl
python visual_score.py gold base finetuned
```

## Notes

- Rendering blocks all network requests, so pages render the same way every time. Remote images and web fonts don't load, which costs every model a little SSIM equally.
- The train/validation split is by row, so pages from the same source repository can appear in both. Validation loss is used to pick a checkpoint. The real test is Flame-Eval-React.
