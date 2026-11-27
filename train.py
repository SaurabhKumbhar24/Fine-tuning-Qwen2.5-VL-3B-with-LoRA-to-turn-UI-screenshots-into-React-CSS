import os
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from datasets import load_from_disk
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from prepare import save_data
from torch.optim import AdamW

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "outputs/qwen2.5vl-lora/"
MAX_PIXELS = 1280 * 28 * 28
MAX_LENGTH = 4096
BATCH_SIZE = 1          # 1 = no padding inside a batch (padding gave nan gradients)
VAL_ROWS = 500          # validation rows; the full 8,300 took far too long
NUM_WORKERS = 8 if os.name != "nt" else 0   # CPU processes preparing batches (Linux only)

# The tokenizer is used in the main process before the DataLoader forks its workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)

def count_tokens(row):
    assistant_text = row["messages"][2]["content"][0]["text"]
    tokens = processor.tokenizer(text=assistant_text)["input_ids"]
    width, height = row["images"][0].size
    img_tokens = processor.image_processor.get_number_of_image_patches(height,width,{}) // 4
    return len(tokens) + img_tokens + 114

def keep_row(row):
    return count_tokens(row) <= MAX_LENGTH

if(not os.path.exists("data/filtered_flame")):
    if(not os.path.exists("data/flame_clean")):
        save_data()
    ds = load_from_disk("data/flame_clean")
    ds = ds.filter(keep_row)
    ds.save_to_disk("data/filtered_flame")
else:
    ds = load_from_disk("data/filtered_flame")


def collate_fn(examples):
    full_text = [processor.apply_chat_template(ex['messages'], tokenize = False) for ex in examples]
    prompt_texts = [
        processor.apply_chat_template(ex['messages'][:2], tokenize=False, add_generation_prompt=True) 
        for ex in examples
    ]
    images = [ex["images"] for ex in examples]

    batch = processor(text=full_text,images=images, padding=True, return_tensors='pt')
    prompt_batch = processor(text=prompt_texts,images=images, padding=True, return_tensors='pt')
    prompt_lens = prompt_batch["attention_mask"].sum(dim=1)
    labels = batch['input_ids'].clone()
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100
    labels[batch["attention_mask"] == 0] = -100

    batch["labels"] = labels

    return batch

train_loader = DataLoader(
    dataset=ds['train'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=NUM_WORKERS > 0
)

val_loader = DataLoader(
    dataset=ds['test'].select(range(VAL_ROWS)),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=NUM_WORKERS > 0
)

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa")

model.to(device)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

model = get_peft_model(model, peft_config=lora_config)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.enable_input_require_grads()

GRAD_ACCUM = 16         # batches per optimizer step (16 x batch 1 = 16 examples)
NUM_EPOCHS = 1          # passes over the data
WARMUP_RATIO = 0.05     # first 5% of steps are warmup
LOG_EVERY = 10          # print training loss every N optimizer steps
EVAL_EVERY = 500        # run validation every N optimizer steps
VAL_PRINT_EVERY = 50    # progress line every N validation batches

trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(
    trainable_params,
    lr = 1e-4,
    weight_decay=0.01
)

total_steps = len(train_loader) // GRAD_ACCUM * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Train batches: {len(train_loader)} | Optimizer steps: {total_steps} | "
      f"Warmup: {warmup_steps} | Val batches: {len(val_loader)} | Workers: {NUM_WORKERS}", flush=True)


def grads_are_finite():
    norms = [torch.linalg.vector_norm(p.grad) for p in trainable_params if p.grad is not None]
    return bool(torch.isfinite(torch.stack(norms).sum()))


def run_validation():
    model.eval()
    total_val_loss = 0
    val_batches = 0
    start = time.time()
    print(f"Validation started ({len(val_loader)} batches)", flush=True)

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            val_batch = val_batch.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                val_loss = model(**val_batch).loss

            # same guard: one nan batch would make the whole average nan
            if torch.isfinite(val_loss):
                total_val_loss += val_loss.item()
                val_batches += 1

            if (i + 1) % VAL_PRINT_EVERY == 0:
                print(f"  validation batch {i + 1}/{len(val_loader)} "
                      f"({time.time() - start:.0f}s)", flush=True)

    model.train()
    skipped = len(val_loader) - val_batches
    avg_val_loss = total_val_loss / max(val_batches, 1)
    print(f"Validation Loss: {avg_val_loss:.4f} (skipped {skipped} non-finite batches, "
          f"{time.time() - start:.0f}s)", flush=True)
    return avg_val_loss


global_step = 0
running_loss = 0.
best_val_loss = float('inf')
skipped_batches = 0

model.train()
optimizer.zero_grad()
last_log_time = time.time()

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(**batch).loss

        batch_info = (f"seq_len={batch['input_ids'].shape[1]} "
                      f"answer_tokens={(batch['labels'] != -100).sum().item()} "
                      f"image_grid_thw={batch['image_grid_thw'].tolist()}")

        # A single nan/inf loss would turn every LoRA weight into nan, so skip it
        if torch.isfinite(loss):
            # keep a copy of the gradients so far (~30 MB), in case this batch adds nan
            grad_backup = [p.grad.clone() if p.grad is not None else None for p in trainable_params]
            (loss/GRAD_ACCUM).backward()

            if grads_are_finite():
                running_loss += loss.item()/GRAD_ACCUM
            else:
                # undo only this batch; the other batches of the group are kept
                for p, g in zip(trainable_params, grad_backup):
                    p.grad = g
                skipped_batches += 1
                print(f"Skipping batch {step}: nan/inf gradients, {batch_info}", flush=True)
            del grad_backup
        else:
            skipped_batches += 1
            print(f"Skipping batch {step}: loss={loss.item()}, {batch_info}", flush=True)

        if (step + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.)
            # last safety net: drop the update instead of writing nan into the weights
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f"Skipping update at step {global_step}: grad norm is {grad_norm}", flush=True)
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                lr = scheduler.get_last_lr()[0]
                seconds = (time.time() - last_log_time) / LOG_EVERY
                hours_left = seconds * (total_steps - global_step) / 3600
                print(f"Epoch: {epoch} | Step {global_step}/{total_steps} | Avg Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | Grad norm: {grad_norm:.2f} | {seconds:.1f}s/step | "
                      f"~{hours_left:.1f}h left (without validation) | skipped batches: {skipped_batches}",
                      flush=True)
                running_loss = 0.
                last_log_time = time.time()

            if global_step % EVAL_EVERY == 0:
                avg_val_loss = run_validation()
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_pretrained(f"{OUTPUT_DIR}checkpoint_{global_step}/")
                    processor.save_pretrained(f"{OUTPUT_DIR}checkpoint_{global_step}/")
                    print(f"Saved new best checkpoint_{global_step}", flush=True)
                last_log_time = time.time()   # validation time is not training time

# final validation, so the end of training is measured too
run_validation()
model.save_pretrained(f"{OUTPUT_DIR}final/")
processor.save_pretrained(f"{OUTPUT_DIR}final/")
print("Training finished, saved final/", flush=True)