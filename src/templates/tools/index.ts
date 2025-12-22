/**
 * ML Tool Cell Generators
 *
 * Generates code cells for optional ML tool integrations.
 */

import type { ToolId, ToolConfig } from '../../types/tools'
import { DEFAULT_TOOL_CONFIG, getToolDependencies } from '../../types/tools'

/**
 * Generate tool cell content
 */
export function generateToolCell(
  toolId: ToolId,
  config?: Partial<ToolConfig>
): string {
  const mergedConfig = {
    ...DEFAULT_TOOL_CONFIG,
    ...config,
  }

  switch (toolId) {
    case 'sacrebleu':
      return generateSacrebleuCell(mergedConfig.sacrebleu!)
    case 'qlora':
      return generateQLoRACell(mergedConfig.qlora!)
    case 'accelerate':
      return generateAccelerateCell(mergedConfig.accelerate!)
    case 'onnx':
      return generateONNXCell(mergedConfig.onnx!)
    case 'streaming':
      return generateStreamingCell(mergedConfig.streaming!)
    case 'hpo':
      return generateHPOCell(mergedConfig.hpo!)
    default:
      throw new Error(`Unknown tool: ${toolId}`)
  }
}

/**
 * Generate dependencies installation cell for selected tools
 */
export function generateToolDependenciesCell(tools: ToolId[]): string {
  const deps = getToolDependencies(tools)
  if (deps.length === 0) return ''

  const depsList = deps.map(d => `        "${d}",`).join('\n')

  return `# %% [markdown]
# # Tool Dependencies

# %%
# Install ML tool dependencies
import subprocess
import sys

def install_tool_packages():
    """Install optional ML tool packages."""
    packages = [
${depsList}
    ]
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {pkg}: {e}")

install_tool_packages()
`
}

// ============================================================================
// SacreBLEU Evaluation
// ============================================================================

function generateSacrebleuCell(config: NonNullable<ToolConfig['sacrebleu']>): string {
  const metrics = config.metrics || ['bleu', 'chrf']
  const tokenizer = config.tokenizer || '13a'
  const lowercase = config.lowercase ? 'True' : 'False'

  return `# %% [markdown]
# # Evaluation with SacreBLEU
#
# Reproducible MT evaluation with standardized metrics and version signatures.

# %%
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

# Evaluation configuration
EVAL_CONFIG = {
    "tokenizer": "${tokenizer}",
    "lowercase": ${lowercase},
    "metrics": ${JSON.stringify(metrics)},
}

def evaluate_translations(
    hypotheses: list[str],
    references: list[str],
    source_texts: list[str] = None,
) -> dict:
    """
    Evaluate translations with SacreBLEU metrics.

    Returns dict with scores and reproducible signatures.
    """
    results = {}

    # Wrap references in list (sacrebleu expects list of reference lists)
    refs = [references]

    ${metrics.includes('bleu') ? `
    # BLEU score
    bleu = BLEU(
        lowercase=${lowercase},
        tokenize="${tokenizer}",
    )
    bleu_result = bleu.corpus_score(hypotheses, refs)
    results["bleu"] = {
        "score": bleu_result.score,
        "signature": bleu_result.signature,
        "precisions": bleu_result.precisions,
        "bp": bleu_result.bp,
        "sys_len": bleu_result.sys_len,
        "ref_len": bleu_result.ref_len,
    }
    print(f"BLEU: {bleu_result.score:.2f}")
    print(f"  Signature: {bleu_result.signature}")
    ` : ''}

    ${metrics.includes('chrf') ? `
    # chrF score (character-level, good for morphologically rich languages)
    chrf = CHRF()
    chrf_result = chrf.corpus_score(hypotheses, refs)
    results["chrf"] = {
        "score": chrf_result.score,
        "signature": chrf_result.signature,
    }
    print(f"chrF: {chrf_result.score:.2f}")
    print(f"  Signature: {chrf_result.signature}")
    ` : ''}

    ${metrics.includes('ter') ? `
    # TER score (Translation Edit Rate)
    ter = TER()
    ter_result = ter.corpus_score(hypotheses, refs)
    results["ter"] = {
        "score": ter_result.score,
        "signature": ter_result.signature,
    }
    print(f"TER: {ter_result.score:.2f}")
    print(f"  Signature: {ter_result.signature}")
    ` : ''}

    return results

def evaluate_with_breakdown(
    hypotheses: list[str],
    references: list[str],
    source_texts: list[str] = None,
) -> dict:
    """
    Detailed evaluation with sentence-level breakdown.
    """
    results = evaluate_translations(hypotheses, references, source_texts)

    # Sentence-level BLEU for error analysis
    sentence_scores = []
    bleu = BLEU(lowercase=${lowercase}, tokenize="${tokenizer}")

    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        sent_bleu = bleu.sentence_score(hyp, [ref])
        sentence_scores.append({
            "index": i,
            "bleu": sent_bleu.score,
            "source": source_texts[i] if source_texts else None,
            "hypothesis": hyp,
            "reference": ref,
        })

    # Sort by BLEU score to find worst translations
    sentence_scores.sort(key=lambda x: x["bleu"])

    results["sentence_scores"] = sentence_scores
    results["worst_10"] = sentence_scores[:10]
    results["best_10"] = sentence_scores[-10:]

    print(f"\\nWorst 5 translations by BLEU:")
    for item in sentence_scores[:5]:
        print(f"  [{item['index']}] BLEU={item['bleu']:.1f}")
        print(f"    Hyp: {item['hypothesis'][:80]}...")
        print(f"    Ref: {item['reference'][:80]}...")

    return results

# Example usage (uncomment to run):
# results = evaluate_translations(predictions, references)
# detailed = evaluate_with_breakdown(predictions, references, sources)
`
}

// ============================================================================
// QLoRA (4-bit Quantization + LoRA)
// ============================================================================

function generateQLoRACell(config: NonNullable<ToolConfig['qlora']>): string {
  const bits = config.bits || 4
  const quantType = config.quant_type || 'nf4'
  const doubleQuant = config.double_quant ? 'True' : 'False'
  const loraR = config.lora_r || 16
  const loraAlpha = config.lora_alpha || 32
  const loraDropout = config.lora_dropout || 0.05
  const targetModules = Array.isArray(config.target_modules)
    ? JSON.stringify(config.target_modules)
    : `"${config.target_modules}"`

  return `# %% [markdown]
# # QLoRA Configuration
#
# Memory-efficient training with ${bits}-bit quantization and LoRA adapters.
# This allows training large models on limited GPU memory.

# %%
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Quantization configuration
QLORA_CONFIG = {
    "bits": ${bits},
    "quant_type": "${quantType}",
    "double_quant": ${doubleQuant},
    "lora_r": ${loraR},
    "lora_alpha": ${loraAlpha},
    "lora_dropout": ${loraDropout},
    "target_modules": ${targetModules},
}

def create_bnb_config():
    """Create BitsAndBytes quantization config."""
    return BitsAndBytesConfig(
        load_in_${bits}bit=True,
        bnb_${bits}bit_quant_type="${quantType}",
        bnb_${bits}bit_compute_dtype=torch.bfloat16,
        bnb_${bits}bit_use_double_quant=${doubleQuant},
    )

def create_lora_config(task_type=TaskType.SEQ_2_SEQ_LM):
    """Create LoRA configuration for PEFT."""
    return LoraConfig(
        r=${loraR},
        lora_alpha=${loraAlpha},
        lora_dropout=${loraDropout},
        target_modules=${targetModules},
        bias="none",
        task_type=task_type,
    )

def load_model_with_qlora(model_name: str, task_type=TaskType.SEQ_2_SEQ_LM):
    """
    Load a model with QLoRA (quantization + LoRA).

    Returns the PEFT model ready for training.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {model_name} with ${bits}-bit quantization...")

    # Load quantized base model
    bnb_config = create_bnb_config()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = create_lora_config(task_type)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def merge_and_save_model(model, tokenizer, output_path: str):
    """
    Merge LoRA adapters into base model and save.

    This creates a standalone model without adapter dependencies.
    """
    print("Merging LoRA adapters into base model...")

    # Merge adapters
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

    # Print model size
    import os
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"Total size: {total_size / 1e9:.2f} GB")

def estimate_memory_usage(model_name: str, batch_size: int = 1):
    """Estimate GPU memory usage for QLoRA training."""
    # Rough estimates based on model size
    param_estimates = {
        "600M": 0.6e9,
        "1.3B": 1.3e9,
        "3.3B": 3.3e9,
    }

    # Find matching estimate
    params = None
    for key, val in param_estimates.items():
        if key.lower() in model_name.lower():
            params = val
            break

    if params is None:
        print("Could not estimate parameters from model name")
        return

    # QLoRA memory estimation
    # ~0.5 bytes per param for 4-bit + overhead
    base_memory = params * 0.5 / 1e9  # GB

    # LoRA adapter memory (very small)
    lora_memory = params * 0.02 * 2 / 1e9  # ~2% of params, FP16

    # Optimizer states (only for LoRA params)
    optimizer_memory = lora_memory * 8  # AdamW: 8x param memory

    # Activation memory (rough estimate)
    activation_memory = batch_size * 0.5  # GB

    total = base_memory + lora_memory + optimizer_memory + activation_memory

    print(f"Estimated memory for {model_name} with batch_size={batch_size}:")
    print(f"  Base model (${bits}-bit): {base_memory:.2f} GB")
    print(f"  LoRA adapters: {lora_memory:.3f} GB")
    print(f"  Optimizer states: {optimizer_memory:.3f} GB")
    print(f"  Activations: {activation_memory:.2f} GB")
    print(f"  Total: {total:.2f} GB")

# Example usage:
# model, tokenizer = load_model_with_qlora(CONFIG["model_name"])
# After training:
# merge_and_save_model(model, tokenizer, f"{OUTPUT_PATH}/merged_model")
`
}

// ============================================================================
// Accelerate Distributed Training
// ============================================================================

function generateAccelerateCell(config: NonNullable<ToolConfig['accelerate']>): string {
  const mixedPrecision = config.mixed_precision || 'fp16'
  const gradientCheckpointing = config.gradient_checkpointing ? 'True' : 'False'
  const gradAccumSteps = config.gradient_accumulation_steps || 4

  return `# %% [markdown]
# # Accelerate Distributed Training
#
# Wrapper for seamless single-GPU to multi-GPU training with mixed precision.

# %%
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torch

# Accelerate configuration
ACCELERATE_CONFIG = {
    "mixed_precision": "${mixedPrecision}",
    "gradient_checkpointing": ${gradientCheckpointing},
    "gradient_accumulation_steps": ${gradAccumSteps},
}

def create_accelerator(seed: int = 42):
    """
    Create and configure Accelerator for distributed training.

    Works seamlessly for:
    - Single GPU
    - Multi-GPU (DDP)
    - Multi-node
    - Mixed precision (FP16/BF16)
    """
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
    )

    accelerator = Accelerator(
        mixed_precision="${mixedPrecision}",
        gradient_accumulation_steps=${gradAccumSteps},
        kwargs_handlers=[ddp_kwargs],
    )

    # Set seed for reproducibility across devices
    set_seed(seed)

    # Print device info
    if accelerator.is_main_process:
        print(f"Accelerator initialized:")
        print(f"  Device: {accelerator.device}")
        print(f"  Num processes: {accelerator.num_processes}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        print(f"  Distributed type: {accelerator.distributed_type}")

    return accelerator

def prepare_for_training(accelerator, model, optimizer, train_dataloader, eval_dataloader=None, scheduler=None):
    """
    Prepare all training components with Accelerator.
    """
    # Enable gradient checkpointing if configured
    if ${gradientCheckpointing} and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")

    # Prepare components
    if scheduler is not None:
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    return model, optimizer, train_dataloader, eval_dataloader, scheduler

def training_step(accelerator, model, batch, optimizer, scheduler=None):
    """
    Execute a single training step with gradient accumulation.
    """
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        # Only step optimizer after accumulation
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return loss.item()

def save_checkpoint(accelerator, model, tokenizer, output_dir: str, step: int):
    """
    Save checkpoint with Accelerator (handles distributed state).
    """
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{output_dir}/checkpoint-{step}",
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(f"{output_dir}/checkpoint-{step}")
        print(f"Checkpoint saved: {output_dir}/checkpoint-{step}")

def distributed_evaluate(accelerator, model, eval_dataloader, compute_metrics_fn):
    """
    Evaluate model across all distributed processes.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

        # Gather predictions from all processes
        predictions = accelerator.gather_for_metrics(predictions)
        labels = accelerator.gather_for_metrics(batch["labels"])

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics on main process only
    if accelerator.is_main_process:
        metrics = compute_metrics_fn(all_predictions, all_labels)
        return metrics

    return None

# Example training loop:
# accelerator = create_accelerator()
# model, optimizer, train_dl, eval_dl, scheduler = prepare_for_training(
#     accelerator, model, optimizer, train_dataloader, eval_dataloader, scheduler
# )
# for epoch in range(num_epochs):
#     for batch in train_dl:
#         loss = training_step(accelerator, model, batch, optimizer, scheduler)
`
}

// ============================================================================
// ONNX Export
// ============================================================================

function generateONNXCell(config: NonNullable<ToolConfig['onnx']>): string {
  const optimizationLevel = config.optimization_level || 'O2'
  const quantize = config.quantize ? 'True' : 'False'

  return `# %% [markdown]
# # ONNX Export for Production Inference
#
# Export and optimize models using Hugging Face Optimum with ONNX Runtime.

# %%
import os
from pathlib import Path

# ONNX export configuration
ONNX_CONFIG = {
    "optimization_level": "${optimizationLevel}",
    "quantize": ${quantize},
}

def export_to_onnx(
    model_path: str,
    output_path: str,
    task: str = "translation",
    optimization_level: str = "${optimizationLevel}",
):
    """
    Export a HuggingFace model to optimized ONNX format.

    Optimization levels:
    - O1: Basic general optimizations
    - O2: Basic + extended + transformer fusions (recommended)
    - O3: O2 + GELU approximation
    - O4: O3 + FP16 mixed precision (GPU only)
    """
    from optimum.exporters.onnx import main_export
    from optimum.onnxruntime import ORTOptimizer
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

    print(f"Exporting {model_path} to ONNX...")
    print(f"  Task: {task}")
    print(f"  Optimization: {optimization_level}")

    # Export to ONNX
    main_export(
        model_name_or_path=model_path,
        output=output_path,
        task=task,
        trust_remote_code=True,
    )

    # Apply optimizations
    if optimization_level != "O0":
        print(f"Applying {optimization_level} optimizations...")
        optimizer = ORTOptimizer.from_pretrained(output_path)
        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level)
        optimizer.optimize(save_dir=output_path, optimization_config=optimization_config)

    # Print output files
    print(f"\\nONNX files saved to {output_path}:")
    for f in os.listdir(output_path):
        fpath = os.path.join(output_path, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  - {f}: {size_mb:.1f} MB")

def load_onnx_model(model_path: str, provider: str = "CUDAExecutionProvider"):
    """
    Load an ONNX model for inference.

    Providers:
    - CUDAExecutionProvider: GPU inference
    - CPUExecutionProvider: CPU inference
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    # Determine provider based on availability
    import onnxruntime as ort
    available_providers = ort.get_available_providers()

    if provider not in available_providers:
        print(f"Warning: {provider} not available, using CPU")
        provider = "CPUExecutionProvider"

    print(f"Loading ONNX model from {model_path}")
    print(f"  Provider: {provider}")

    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_path,
        provider=provider,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def benchmark_onnx_inference(
    model,
    tokenizer,
    test_texts: list[str],
    batch_size: int = 8,
    num_runs: int = 3,
):
    """
    Benchmark ONNX model inference speed.
    """
    import time
    import numpy as np

    # Warmup
    print("Warming up...")
    inputs = tokenizer(test_texts[:batch_size], return_tensors="pt", padding=True)
    _ = model.generate(**inputs, max_length=128)

    # Benchmark
    times = []
    total_tokens = 0

    print(f"Running {num_runs} benchmark iterations...")
    for run in range(num_runs):
        start = time.perf_counter()

        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs, max_length=128)
            total_tokens += outputs.shape[0] * outputs.shape[1]

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = total_tokens / (avg_time * num_runs)

    print(f"\\nBenchmark results:")
    print(f"  Average time: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Samples: {len(test_texts)}")

    return {"avg_time": avg_time, "std_time": std_time, "tokens_per_sec": tokens_per_sec}

def quantize_onnx_model(model_path: str, output_path: str = None):
    """
    Apply dynamic INT8 quantization to ONNX model.

    Reduces model size and can improve inference speed on CPU.
    """
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if output_path is None:
        output_path = model_path + "_quantized"

    print(f"Quantizing ONNX model to INT8...")

    quantizer = ORTQuantizer.from_pretrained(model_path)
    quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False)

    quantizer.quantize(
        save_dir=output_path,
        quantization_config=quantization_config,
    )

    print(f"Quantized model saved to {output_path}")

# Example usage:
# export_to_onnx(MODEL_PATH, f"{OUTPUT_PATH}/onnx_model")
# model, tokenizer = load_onnx_model(f"{OUTPUT_PATH}/onnx_model")
# benchmark_onnx_inference(model, tokenizer, test_texts)
`
}

// ============================================================================
// Streaming DataLoader
// ============================================================================

function generateStreamingCell(config: NonNullable<ToolConfig['streaming']>): string {
  const bufferSize = config.buffer_size || 10000
  const numWorkers = config.num_workers || 4

  return `# %% [markdown]
# # Streaming DataLoader
#
# Memory-efficient data loading for large datasets that don't fit in RAM.

# %%
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
import torch

# Streaming configuration
STREAMING_CONFIG = {
    "buffer_size": ${bufferSize},
    "num_workers": ${numWorkers},
}

def load_streaming_dataset(
    data_path: str,
    split: str = "train",
    streaming: bool = True,
):
    """
    Load dataset in streaming mode for memory efficiency.

    Works with:
    - Hugging Face Hub datasets
    - Local CSV/JSON/Parquet files
    - Remote URLs
    """
    print(f"Loading dataset: {data_path}")
    print(f"  Split: {split}")
    print(f"  Streaming: {streaming}")

    # Detect file type
    if data_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_path, split=split, streaming=streaming)
    elif data_path.endswith('.json') or data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split=split, streaming=streaming)
    elif data_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=data_path, split=split, streaming=streaming)
    else:
        # Assume HuggingFace dataset
        dataset = load_dataset(data_path, split=split, streaming=streaming)

    return dataset

def create_streaming_dataloader(
    dataset: IterableDataset,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    src_column: str = "source",
    tgt_column: str = "target",
    shuffle_buffer: int = ${bufferSize},
):
    """
    Create a DataLoader from a streaming dataset with on-the-fly tokenization.
    """

    def tokenize_batch(examples):
        """Tokenize a batch of examples."""
        inputs = tokenizer(
            examples[src_column],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[tgt_column],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        inputs["labels"] = labels["input_ids"]
        return inputs

    def collate_fn(batch):
        """Collate batch of examples."""
        # Stack all tensors
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in batch])
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in batch])
        labels = torch.stack([torch.tensor(ex["labels"]) for ex in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Shuffle if buffer specified
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Apply tokenization
    dataset = dataset.map(
        lambda ex: {
            **tokenizer(ex[src_column], max_length=max_length, truncation=True, padding="max_length"),
            "labels": tokenizer(ex[tgt_column], max_length=max_length, truncation=True, padding="max_length")["input_ids"],
        },
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=${numWorkers},
        pin_memory=True,
    )

    return dataloader

def create_distributed_streaming_dataloader(
    dataset: IterableDataset,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    src_column: str = "source",
    tgt_column: str = "target",
    world_size: int = 1,
    rank: int = 0,
):
    """
    Create a streaming DataLoader for distributed training.

    Each process gets a different shard of the data.
    """
    from datasets.distributed import split_dataset_by_node

    # Split dataset across processes
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    # Create dataloader (same as above)
    return create_streaming_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        src_column=src_column,
        tgt_column=tgt_column,
        shuffle_buffer=${bufferSize} // world_size,
    )

def estimate_dataset_size(dataset: IterableDataset, sample_size: int = 1000):
    """
    Estimate total dataset size by sampling.
    """
    import sys

    total_bytes = 0
    count = 0

    for example in dataset.take(sample_size):
        total_bytes += sys.getsizeof(str(example))
        count += 1

    if count == 0:
        return None

    avg_size = total_bytes / count
    print(f"Sampled {count} examples")
    print(f"  Average size: {avg_size:.1f} bytes/example")
    print(f"  Note: Total size depends on dataset length")

    return avg_size

# Example usage:
# dataset = load_streaming_dataset("path/to/data.csv")
# dataloader = create_streaming_dataloader(dataset, tokenizer, batch_size=8)
# for batch in dataloader:
#     outputs = model(**batch)
`
}

// ============================================================================
// Hyperparameter Optimization
// ============================================================================

function generateHPOCell(config: NonNullable<ToolConfig['hpo']>): string {
  const nTrials = config.n_trials || 20
  const metric = config.metric || 'eval_bleu'
  const direction = config.direction || 'maximize'
  const searchSpace = config.search_space || {}

  return `# %% [markdown]
# # Hyperparameter Optimization
#
# Automated hyperparameter search with Optuna and MLflow tracking.

# %%
import optuna
from optuna.integration import MLflowCallback
import mlflow
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# HPO configuration
HPO_CONFIG = {
    "n_trials": ${nTrials},
    "metric": "${metric}",
    "direction": "${direction}",
    "study_name": "akkadian-hpo",
    "storage": "sqlite:///optuna_study.db",  # Persistent storage
}

def create_hp_space(trial):
    """
    Define hyperparameter search space.

    Uses Optuna's suggest_* methods for sampling.
    """
    return {
        ${searchSpace.learning_rate ? `"learning_rate": trial.suggest_float("learning_rate", ${searchSpace.learning_rate.min}, ${searchSpace.learning_rate.max}, log=${searchSpace.learning_rate.log ? 'True' : 'False'}),` : '# "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),'}
        ${searchSpace.batch_size ? `"per_device_train_batch_size": trial.suggest_categorical("batch_size", ${JSON.stringify(searchSpace.batch_size)}),` : '# "per_device_train_batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),'}
        ${searchSpace.weight_decay ? `"weight_decay": trial.suggest_float("weight_decay", ${searchSpace.weight_decay.min}, ${searchSpace.weight_decay.max}),` : '# "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),'}
        ${searchSpace.warmup_ratio ? `"warmup_ratio": trial.suggest_float("warmup_ratio", ${searchSpace.warmup_ratio.min}, ${searchSpace.warmup_ratio.max}),` : '# "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),'}
        ${searchSpace.lora_r ? `"lora_r": trial.suggest_categorical("lora_r", ${JSON.stringify(searchSpace.lora_r)}),` : ''}
        ${searchSpace.lora_alpha ? `"lora_alpha": trial.suggest_categorical("lora_alpha", ${JSON.stringify(searchSpace.lora_alpha)}),` : ''}
    }

def run_hpo_with_trainer(
    model_init_fn,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
    n_trials: int = ${nTrials},
    metric: str = "${metric}",
    direction: str = "${direction}",
):
    """
    Run hyperparameter optimization using HuggingFace Trainer.

    Each trial is logged as a child run in MLflow.
    """

    # Start MLflow parent run
    with mlflow.start_run(run_name="hpo_study") as parent_run:
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("metric", metric)
        mlflow.log_param("direction", direction)

        def model_init():
            return model_init_fn()

        # Create trainer with HPO
        training_args = TrainingArguments(
            output_dir=CHECKPOINT_PATH,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric,
            greater_is_better=(direction == "maximize"),
            report_to="mlflow",
            logging_dir=f"{OUTPUT_PATH}/logs",
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Run hyperparameter search
        best_trial = trainer.hyperparameter_search(
            direction=direction,
            backend="optuna",
            hp_space=create_hp_space,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics[metric],
        )

        # Log best trial
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.hyperparameters.items()})
        mlflow.log_metric(f"best_{metric}", best_trial.objective)

        print(f"\\nBest trial:")
        print(f"  {metric}: {best_trial.objective}")
        print(f"  Hyperparameters: {best_trial.hyperparameters}")

        return best_trial, trainer

def run_hpo_manual(
    objective_fn,
    n_trials: int = ${nTrials},
    direction: str = "${direction}",
    study_name: str = "akkadian-hpo",
    storage: str = "sqlite:///optuna_study.db",
):
    """
    Run HPO with manual Optuna study for more control.

    The objective function should:
    1. Accept a trial object
    2. Sample hyperparameters using trial.suggest_*
    3. Train and evaluate the model
    4. Return the metric value
    """

    # Create or load study with persistent storage
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )

    # MLflow callback for logging
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="${metric}",
        create_experiment=False,
    )

    # Run optimization
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )

    # Print results
    print(f"\\nBest trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Log summary to MLflow
    with mlflow.start_run(run_name="hpo_summary"):
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metric("best_${metric}", study.best_trial.value)
        mlflow.log_metric("n_trials", len(study.trials))

    return study

def visualize_hpo_results(study):
    """
    Generate visualization plots for HPO results.
    """
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour,
    )

    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_html(f"{OUTPUT_PATH}/hpo_history.html")
    print(f"Saved: {OUTPUT_PATH}/hpo_history.html")

    # Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.write_html(f"{OUTPUT_PATH}/hpo_importances.html")
        print(f"Saved: {OUTPUT_PATH}/hpo_importances.html")
    except Exception as e:
        print(f"Could not plot importances: {e}")

    # Parallel coordinate plot
    fig = plot_parallel_coordinate(study)
    fig.write_html(f"{OUTPUT_PATH}/hpo_parallel.html")
    print(f"Saved: {OUTPUT_PATH}/hpo_parallel.html")

def resume_hpo_study(study_name: str = "akkadian-hpo", storage: str = "sqlite:///optuna_study.db"):
    """
    Resume a previous HPO study.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)

    print(f"Loaded study: {study_name}")
    print(f"  Completed trials: {len(study.trials)}")
    print(f"  Best value: {study.best_trial.value}")
    print(f"  Best params: {study.best_trial.params}")

    return study

# Example usage:
# best_trial, trainer = run_hpo_with_trainer(
#     model_init_fn=lambda: load_model(),
#     train_dataset=train_tokenized,
#     eval_dataset=val_tokenized,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )
`
}

// Export all generators
export {
  generateSacrebleuCell,
  generateQLoRACell,
  generateAccelerateCell,
  generateONNXCell,
  generateStreamingCell,
  generateHPOCell,
}
