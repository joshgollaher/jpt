import argparse
import math
import os
from datasets import load_dataset, Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments, pipeline,
)

# -----------------------------------
MODEL_NAME = "jpt"
BLOCK_SIZE = 512
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
USE_DISCORD = False
# ------------------------------------


if USE_DISCORD:
    with open("data/discord.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    dataset = Dataset.from_dict({"text": lines})
else:
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


tokenizer_path = "./tokenizer"
if not os.path.exists(tokenizer_path):
    from tokenizers import ByteLevelBPETokenizer

    os.makedirs(tokenizer_path, exist_ok=True)
    # Dump texts to train tokenizer
    with open("wikitext.txt", "w", encoding="utf-8") as f:
        f.writelines(dataset["text"])
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["wikitext.txt"], vocab_size=50257, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    tokenizer.save_model(tokenizer_path)

tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

tokenized = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])


def group_texts(examples):
    joined = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(joined["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in joined.items()
    }
    return result


lm_dataset = tokenized.map(group_texts, batched=True, num_proc=26)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=BLOCK_SIZE,
    n_ctx=BLOCK_SIZE,
    n_embd=768,
    n_layer=12,
    n_head=12,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)


training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    eval_dataset=lm_dataset.select(range(1000)),
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Whether to train or test the model')
    args = parser.parse_args()

    if args.mode == 'train':
        trainer.train()
    else:
        pipe = pipeline("text-generation", model="./checkpoints", tokenizer=tokenizer)
        print(pipe("Tell me a cool fact: ", max_new_tokens=50)[0]["generated_text"])

