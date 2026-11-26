import os
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
BATCH_SIZE = 2

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
    shuffle=True
)

val_loader = DataLoader(
    dataset=ds['test'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=False
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

GRAD_ACCUM = 8          # batches per optimizer step
NUM_EPOCHS = 1          # passes over the data
WARMUP_RATIO = 0.05     # first 5% of steps are warmup

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


global_step = 0
running_loss = 0.
best_val_loss = float('inf')

model.train()
optimizer.zero_grad()

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(**batch).loss

        (loss/GRAD_ACCUM).backward()
        running_loss += loss.item()/GRAD_ACCUM
        
        if (step + 1) % GRAD_ACCUM == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if(global_step % 10 == 0):
                avg_loss = running_loss / 10
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch: {epoch} | Step {global_step}/{total_steps} | Avg Loss: {avg_loss} | LR: {lr: .2e} | Grad norm: {grad_norm: .2f}")
                running_loss = 0.

            if(global_step % 100 == 0):

                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for _, val_batch in enumerate(val_loader):
                        val_batch = val_batch.to(device)

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            val_loss = model(**val_batch).loss

                        total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss/len(val_loader)

                print(f"Validation Loss: {avg_val_loss}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_pretrained(f"{OUTPUT_DIR}checkpoint_{global_step}/")
                    processor.save_pretrained(f"{OUTPUT_DIR}checkpoint_{global_step}/")
                
                model.train()

model.save_pretrained(f"{OUTPUT_DIR}final/")
processor.save_pretrained(f"{OUTPUT_DIR}final/")