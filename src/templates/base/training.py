# @@TEMPLATE: training
# @@DESCRIPTION: Base training template for sequence-to-sequence translation models
# @@VERSION: 1.0.0
# @@PLATFORMS: kaggle-p100, kaggle-t4x2, colab-free, colab-pro, vertex-a100, runpod-a100

# --- INJECTION POINT: platform_setup ---
# Platform-specific setup will be injected here
# {{PLATFORM_SETUP}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Training Configuration

# %%
# Training Configuration
CONFIG = {
    "model_name": "{{model_name}}",
    "src_lang": "{{src_lang}}",
    "tgt_lang": "{{tgt_lang}}",
    "num_epochs": {{num_epochs}},
    "batch_size": {{batch_size}},
    "gradient_accumulation_steps": {{gradient_accumulation_steps}},
    "learning_rate": {{learning_rate}},
    "max_src_len": {{max_src_len}},
    "max_tgt_len": {{max_tgt_len}},
    "warmup_ratio": {{warmup_ratio}},
    "weight_decay": {{weight_decay}},
    "fp16": {{fp16}},
    "save_steps": {{save_steps}},
    "eval_steps": {{eval_steps}},
    "logging_steps": {{logging_steps}},
    "save_total_limit": {{save_total_limit}},
}

print("Training Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# # Dependencies

# %%
# Install dependencies
import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate",
        "sentencepiece",
        "sacrebleu",
        "evaluate",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

# %%
# Imports
import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- INJECTION POINT: data_loading ---
# Platform-specific data loading will be injected here
# {{DATA_LOADING}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Data Preprocessing

# %%
# Create train/val split if no validation set
if val_df is None:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Split: {len(train_df)} train, {len(val_df)} validation")

# Detect source and target columns
src_col = None
tgt_col = None
for col in train_df.columns:
    col_lower = col.lower()
    if 'source' in col_lower or 'src' in col_lower or 'input' in col_lower:
        src_col = col
    elif 'target' in col_lower or 'tgt' in col_lower or 'output' in col_lower or 'english' in col_lower:
        tgt_col = col

if src_col is None or tgt_col is None:
    # Fall back to first two columns
    src_col, tgt_col = train_df.columns[:2]

print(f"Source column: {src_col}")
print(f"Target column: {tgt_col}")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[[src_col, tgt_col]].rename(
    columns={src_col: 'source', tgt_col: 'target'}
))
val_dataset = Dataset.from_pandas(val_df[[src_col, tgt_col]].rename(
    columns={src_col: 'source', tgt_col: 'target'}
))

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")

# %% [markdown]
# # Model and Tokenizer

# %%
# Load model and tokenizer
print(f"Loading model: {CONFIG['model_name']}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

# Set source/target languages for multilingual models
if hasattr(tokenizer, 'src_lang'):
    tokenizer.src_lang = CONFIG["src_lang"]
if hasattr(tokenizer, 'tgt_lang'):
    tokenizer.tgt_lang = CONFIG["tgt_lang"]

print(f"Model parameters: {model.num_parameters():,}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

# %%
# Preprocessing function
def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_src_len"],
        truncation=True,
        padding=False,
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=CONFIG["max_tgt_len"],
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)
val_tokenized = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val",
)

# %% [markdown]
# # Training Setup

# %%
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8 if CONFIG["fp16"] else None,
)

# Metrics
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[ref] for ref in decoded_labels]
    )

    return {"bleu": result["score"]}

# %%
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_PATH,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    fp16=CONFIG["fp16"],
    evaluation_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    logging_steps=CONFIG["logging_steps"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,  # Faster training without generation
    dataloader_num_workers=0,
    report_to="none",
)

# %%
# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# %% [markdown]
# # Training

# %%
# Clear GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train
print("Starting training...")
train_result = trainer.train()

# Log results
print(f"\nTraining completed!")
print(f"Training loss: {train_result.training_loss:.4f}")

# --- INJECTION POINT: checkpoint_save ---
# Platform-specific checkpoint saving will be injected here
# {{CHECKPOINT_SAVE}}
# --- END INJECTION POINT ---

# %% [markdown]
# # Evaluation

# %%
# Final evaluation with generation
print("\nRunning final evaluation with BLEU...")

# Update args for generation
trainer.args.predict_with_generate = True
trainer.args.generation_max_length = CONFIG["max_tgt_len"]

eval_results = trainer.evaluate()
print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
if 'eval_bleu' in eval_results:
    print(f"Final BLEU: {eval_results['eval_bleu']:.2f}")

# --- INJECTION POINT: output_save ---
# Platform-specific output saving will be injected here
# {{OUTPUT_SAVE}}
# --- END INJECTION POINT ---

# %%
# Save final model
print("\nSaving final model...")
save_model_for_submission(model, tokenizer, output_name="final_model")

# Save training metrics
training_history = {
    "config": CONFIG,
    "train_loss": train_result.training_loss,
    "eval_results": eval_results,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
}
save_training_log(training_history)

print("\nTraining complete!")
