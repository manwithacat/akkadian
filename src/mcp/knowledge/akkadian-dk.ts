/**
 * Akkadian Domain Knowledge
 *
 * Comprehensive reference for the Akkadian ML CLI.
 * Designed for token-efficient loading by LLM agents.
 */

import type { DomainKnowledge } from './types'

export const akkadianDK: DomainKnowledge = {
  version: '1.0.0',
  name: 'Akkadian ML CLI',
  description: 'CLI for ML workflows: Kaggle competitions, Colab training, MLflow tracking',

  quick: {
    purpose:
      'Manage ML training workflows across Kaggle, Colab, Vertex AI, and local environments with unified versioning and tracking.',
    commands: [
      'akk doctor - Check environment and dependencies',
      'akk kaggle upload-notebook <path> - Upload notebook with auto-versioning',
      'akk kaggle status <slug> - Check kernel execution status',
      'akk kaggle submissions - List competition submissions and scores',
      'akk kaggle list-kernels - List kernel versions',
      'akk kaggle logs <slug> - Get kernel execution logs',
      'akk kaggle list-running - List currently running/queued kernels',
      'akk kaggle create-inference --model <handle> - Generate submission kernel',
      'akk colab upload-notebook <path> - Upload to GCS for Colab',
      'akk colab status --run <name> - Check training status',
      'akk colab download-artifacts --run <name> - Download results',
      'akk template generate training --platform kaggle - Generate notebook',
      'akk template list - List templates and platforms',
      'akk preflight check <notebook> --platform kaggle-p100 - Validate resources',
      'akk preflight validate <notebook> - Full validation suite',
      'akk notebook build <config.toml> - Generate notebook from TOML config',
      'akk mlflow start - Start tracking server',
      'akk local infer --model <path> --interactive - Test translations',
      'akk competition init <slug> - Initialize competition directory',
      'akk competition status - Show competition state',
      'akk data download - Download competition data, convert to SQLite',
      'akk data list - List registered dataset versions',
      'akk data register <path> --name <name> - Register dataset with lineage',
      'akk data explore - Launch Datasette to explore datasets',
    ],
  },

  commands: {
    'kaggle upload-notebook': {
      name: 'kaggle upload-notebook',
      description: 'Upload notebook to Kaggle with automatic versioning',
      usage: 'akk kaggle upload-notebook <path> [options]',
      options: {
        '--strategy': 'Versioning: semver (default), timestamp, experiment, overwrite',
        '--dry-run': 'Preview version without uploading',
        '--model': 'Associate with model name',
        '--notes': 'Version notes',
        '--no-version': 'Skip versioning',
        '--gpu': 'Enable GPU (default: true)',
        '--competition': 'Competition slug',
      },
      examples: [
        'akk kaggle upload-notebook train.py',
        'akk kaggle upload-notebook train.py --strategy timestamp',
        'akk kaggle upload-notebook train.py --model nllb-v4 --notes "Fixed eval"',
      ],
    },
    'kaggle status': {
      name: 'kaggle status',
      description: 'Check kernel execution status (running, complete, error)',
      usage: 'akk kaggle status <user/kernel-name>',
      options: {},
      examples: ['akk kaggle status manwithacat/nllb-train-v1'],
    },
    'kaggle submissions': {
      name: 'kaggle submissions',
      description: 'List competition submissions with scoring status',
      usage: 'akk kaggle submissions [options]',
      options: {
        '--competition': 'Competition slug (default: from config)',
        '--limit': 'Number of submissions (default: 10)',
        '--pending': 'Show only pending (being scored)',
      },
      examples: ['akk kaggle submissions', 'akk kaggle submissions --pending', 'akk kaggle submissions --limit 5'],
    },
    'kaggle logs': {
      name: 'kaggle logs',
      description: 'Retrieve kernel execution logs',
      usage: 'akk kaggle logs <user/kernel-name> [options]',
      options: {
        '--errors-only': 'Show only error messages',
        '--tail': 'Show last N lines',
        '--save': 'Save logs to file',
      },
      examples: [
        'akk kaggle logs manwithacat/nllb-train-v1',
        'akk kaggle logs manwithacat/nllb-train-v1 --errors-only',
      ],
    },
    'kaggle list-running': {
      name: 'kaggle list-running',
      description: 'List currently running or queued kernels on Kaggle',
      usage: 'akk kaggle list-running [options]',
      options: {
        '--user': 'Kaggle username (default: from config)',
      },
      examples: ['akk kaggle list-running', 'akk kaggle list-running --user manwithacat'],
    },
    'kaggle create-inference': {
      name: 'kaggle create-inference',
      description: 'Generate complete inference kernel for competition submission with all constraints met',
      usage: 'akk kaggle create-inference --model <handle> [options]',
      options: {
        '--model': 'Kaggle model handle (owner/model-name or full handle)',
        '--name': 'Inference kernel name (default: derived from model)',
        '--version': 'Version (default: extracted from model name)',
        '--output': 'Output directory',
        '--competition': 'Competition slug (default: from config)',
        '--dry-run': 'Preview without writing files',
      },
      examples: [
        'akk kaggle create-inference --model manwithacat/akkadian-byt5-v1-0-10',
        'akk kaggle create-inference --model manwithacat/byt5-consonantal/transformers/default --name byt5-inference',
        'akk kaggle create-inference --model manwithacat/akkadian-nllb-v1-2-0 --dry-run',
      ],
      notes: [
        'Auto-detects model type (ByT5, NLLB, T5, Qwen, Gemma) and preprocessing requirements',
        'Ensures all competition constraints: internet=OFF, kagglehub model loading, submission.csv output',
        'Generates both notebook.py and kernel-metadata.json ready for kaggle kernels push',
      ],
    },
    'template generate': {
      name: 'template generate',
      description: 'Generate notebook from template with platform-specific setup',
      usage: 'akk template generate <template> --platform <platform> [options]',
      options: {
        '--platform': 'Target: kaggle-p100, kaggle-t4x2, colab-free, colab-pro, vertex-a100',
        '--model': 'Model name for config',
        '--tools': 'Enable tools: sacrebleu, qlora, accelerate, onnx, streaming, hpo',
        '--output': 'Output path',
      },
      examples: [
        'akk template generate training --platform kaggle-p100',
        'akk template generate training --platform colab-pro --tools qlora,accelerate',
      ],
    },
    'preflight check': {
      name: 'preflight check',
      description: 'Check notebook resource requirements against platform limits',
      usage: 'akk preflight check <notebook> --platform <platform>',
      options: {
        '--platform': 'Target platform profile',
        '--model': 'Model for memory estimation',
      },
      examples: [
        'akk preflight check train.py --platform kaggle-p100',
        'akk preflight check train.py --platform kaggle-p100 --model facebook/nllb-200-1.3B',
      ],
    },
    'preflight validate': {
      name: 'preflight validate',
      description: 'Full validation: structure, syntax, config, dependencies, security',
      usage: 'akk preflight validate <notebook> [options]',
      options: {
        '--platform': 'Platform for compatibility check',
        '--strict': 'Fail on warnings',
        '--skip-syntax': 'Skip Python syntax check',
        '--skip-security': 'Skip security scan',
      },
      examples: ['akk preflight validate train.py --platform kaggle-p100'],
    },
    'notebook build': {
      name: 'notebook build',
      description: 'Generate notebook deterministically from TOML config file',
      usage: 'akk notebook build <config.toml> [options]',
      options: {
        '--output': 'Output file path (default: derived from config name)',
        '--dry-run': 'Preview what would be generated without writing',
        '--skip-preflight': 'Skip automatic preflight validation',
      },
      examples: [
        'akk notebook build training.toml',
        'akk notebook build training.toml -o train.py',
        'akk notebook build training.toml --dry-run',
      ],
    },
    'colab status': {
      name: 'colab status',
      description: 'Check training run status from GCS',
      usage: 'akk colab status --run <name> [options]',
      options: {
        '--run': 'Run name to check',
        '--watch': 'Continuously poll for updates',
      },
      examples: ['akk colab status --run nllb-v4', 'akk colab status --run nllb-v4 --watch'],
    },
    'data download': {
      name: 'data download',
      description: 'Download competition data from Kaggle and convert to SQLite',
      usage: 'akk data download [options]',
      options: {
        '--competition': 'Competition slug (default: from akk.toml)',
        '--name': 'Dataset name for registration (default: raw)',
        '--force': 'Overwrite existing files',
        '--skip-sqlite': 'Skip SQLite conversion',
        '--skip-register': 'Skip dataset registration',
      },
      examples: [
        'akk data download',
        'akk data download --competition deep-past-initiative-machine-translation',
        'akk data download --name competition_v1 --force',
      ],
    },
    'data list': {
      name: 'data list',
      description: 'List registered dataset versions with lineage info',
      usage: 'akk data list [options]',
      options: {
        '--name': 'Filter by dataset name',
        '--source': 'Filter by source: all, kaggle, etl, derived',
        '--mlflow': 'Show MLflow run linkages',
        '--verbose': 'Show full metadata',
      },
      examples: ['akk data list', 'akk data list --name raw', 'akk data list --source kaggle --mlflow'],
    },
    'data register': {
      name: 'data register',
      description: 'Register a dataset version with lineage tracking',
      usage: 'akk data register <path> --name <name> [options]',
      options: {
        '--name': 'Dataset name (required)',
        '--parent': 'Parent dataset reference (e.g., raw:1)',
        '--pipeline': 'ETL pipeline that created this',
        '--mlflow-run': 'Link to MLflow run ID',
        '--link-type': 'MLflow link type: training, evaluation, inference',
        '--source': 'Source type: etl (default), derived',
      },
      examples: [
        'akk data register ./output/augmented.csv --name v2_augmented',
        'akk data register ./data.db --name processed --parent raw:1 --pipeline augmentation_v2',
        'akk data register ./data.db --name train_v3 --mlflow-run abc123 --link-type training',
      ],
    },
    'data explore': {
      name: 'data explore',
      description: 'Launch Datasette to explore registered datasets',
      usage: 'akk data explore [options]',
      options: {
        '--name': 'Dataset name to explore',
        '--version': 'Specific version (requires --name)',
        '--port': 'Datasette port (default: 8001)',
        '--no-browser': 'Do not open browser',
      },
      examples: ['akk data explore', 'akk data explore --name raw', 'akk data explore --name v2_augmented --version 1'],
    },
  },

  workflows: [
    {
      id: 'config-driven-training',
      name: 'Config-Driven Training',
      description: 'Generate deterministic notebooks from TOML config files',
      when: 'Starting a new training run or iterating on hyperparameters',
      steps: [
        {
          action: 'Create/edit config',
          notes: 'Edit training.toml with model, data, hyperparameters',
        },
        {
          action: 'Build notebook',
          command: 'akk notebook build training.toml',
        },
        {
          action: 'Preflight validates automatically',
          notes: 'Checks GPU/disk/time against platform limits',
        },
        { action: 'Push to Kaggle', command: 'kaggle kernels push' },
        {
          action: 'Model auto-uploads',
          notes: 'Training notebook uploads to Kaggle Model Registry with rich metadata',
        },
      ],
      benefits: [
        'Single source of truth for all ML decisions',
        'Automatic preflight validation catches errors before remote execution',
        'Semver model naming (akkadian-byt5-v1-0-11) instead of confusing variations',
        'Rich metadata maximizes Kaggle usability score',
      ],
    },
    {
      id: 'competition-submission',
      name: 'Two-Stage Competition Submission',
      description: 'Separate training (internet ON) from inference (internet OFF) for competitions',
      when: 'Submitting to Kaggle competitions with internet restrictions',
      steps: [
        {
          action: 'Training notebook',
          notes: 'internet=ON, downloads from HuggingFace, uploads to Kaggle Model Registry',
        },
        {
          action: 'Monitor training',
          command: 'akk kaggle list-running',
          notes: 'Lists all running/queued kernels sorted by recent activity',
        },
        {
          action: 'Wait for completion',
          command: 'akk kaggle status <training-kernel>',
        },
        {
          action: 'Generate inference kernel',
          command: 'akk kaggle create-inference --model <owner/model-name>',
          notes: 'Auto-generates notebook with all constraints met: internet=OFF, kagglehub loading, submission.csv',
        },
        {
          action: 'Push inference kernel',
          command: 'kaggle kernels push -p <output-dir>',
        },
        {
          action: 'Submit for scoring',
          notes: 'Inference kernel outputs submission.csv, competition scores it',
        },
      ],
      why: [
        'Competition kernels must have internet=OFF to submit',
        'HuggingFace from_pretrained() requires internet',
        'Solution: Train with internet, upload model to registry, infer from registry without internet',
        'akk kaggle create-inference auto-generates compliant notebooks',
      ],
    },
    {
      id: 'kaggle-train',
      name: 'Kaggle Training',
      description: 'Train a model on Kaggle GPU',
      when: 'Training on Kaggle P100 or T4x2',
      steps: [
        {
          action: 'Generate notebook',
          command: 'akk template generate training --platform kaggle-p100',
        },
        {
          action: 'Validate notebook',
          command: 'akk preflight validate train.py --platform kaggle-p100',
        },
        {
          action: 'Upload with versioning',
          command: 'akk kaggle upload-notebook train.py --model nllb-v4',
        },
        {
          action: 'Monitor kernel',
          command: 'akk kaggle run-kernel <slug> --wait',
        },
        {
          action: 'Download outputs',
          command: 'akk kaggle download-output <slug>',
        },
      ],
    },
    {
      id: 'colab-train',
      name: 'Colab Training',
      description: 'Train on Colab with GCS sync',
      when: 'Training on Colab Pro with A100 or longer sessions',
      steps: [
        {
          action: 'Generate notebook',
          command: 'akk template generate training --platform colab-pro',
        },
        {
          action: 'Upload to GCS',
          command: 'akk colab upload-notebook train.py --version v4',
        },
        { action: 'Open in Colab', notes: 'Open from GCS link' },
        {
          action: 'Monitor progress',
          command: 'akk colab status --run nllb-v4 --watch',
        },
        {
          action: 'Download artifacts',
          command: 'akk colab download-artifacts --run nllb-v4',
        },
        {
          action: 'Import to MLflow',
          command: 'akk workflow import-run --run nllb-v4',
        },
      ],
    },
    {
      id: 'improve-score',
      name: 'Improve Model Score',
      description: 'Iterate on model to improve competition score',
      when: 'Current score is not satisfactory',
      steps: [
        {
          action: 'Analyze current results',
          command: 'akk local analyze --run <current>',
        },
        {
          action: 'Compare with baseline',
          command: 'akk local analyze --run <current> --compare <baseline>',
        },
        {
          action: 'Test translations',
          command: 'akk local infer --model ./models/<version> --interactive',
        },
        {
          action: 'Adjust hyperparameters',
          notes: 'Based on analysis recommendations',
        },
        {
          action: 'Train new version',
          notes: 'Use kaggle-train or colab-train workflow',
        },
      ],
    },
    {
      id: 'data-pipeline',
      name: 'Data Pipeline with Lineage',
      description: 'Manage datasets with version and lineage tracking',
      when: 'Starting a new competition or creating derived datasets',
      steps: [
        { action: 'Download competition data', command: 'akk data download' },
        { action: 'Explore raw data', command: 'akk data explore --name raw' },
        {
          action: 'Run ETL pipeline',
          notes: 'Create augmented/cleaned dataset',
        },
        {
          action: 'Register derived dataset',
          command:
            'akk data register ./output/augmented.csv --name v2_augmented --parent raw:1 --pipeline augmentation_v2',
        },
        {
          action: 'Link to training run',
          command: 'akk data register ./data.db --name train_v3 --mlflow-run <run-id> --link-type training',
        },
        { action: 'List all datasets', command: 'akk data list --mlflow' },
      ],
    },
  ],

  patterns: [
    {
      id: 'kernel-vs-submission',
      name: 'Kernel Status vs Submission Scoring',
      problem: 'Kernel complete but no score yet',
      solution:
        'Kernel execution and competition scoring are separate. Use kaggle status for kernel, kaggle submissions for scoring.',
      commands: [
        'akk kaggle status <user/kernel> - Check if notebook finished running',
        'akk kaggle submissions --pending - Check if submission is being scored',
      ],
    },
    {
      id: 'oom-fix',
      name: 'Fix OOM Errors',
      problem: 'Out of memory during training',
      solution: 'Reduce batch size, enable gradient accumulation, use fp16/bf16',
      commands: [
        'akk preflight check train.py --platform kaggle-p100 --model <model>',
        'akk template generate training --platform kaggle-p100 --tools qlora',
      ],
    },
    {
      id: 'version-tracking',
      name: 'Track Notebook Versions',
      problem: 'Kaggle version UI is confusing',
      solution: 'Use akk versioning with local registry and MLflow tracking',
      commands: [
        'akk kaggle upload-notebook train.py --strategy semver --model nllb-v4',
        'akk kaggle list-kernels --name train',
        'akk kaggle list-kernels --mlflow',
      ],
    },
    {
      id: 'cross-account-colab',
      name: 'Cross-Account Colab',
      problem: 'Colab account differs from GCS bucket owner',
      solution: 'Grant objectAdmin to Colab user or use service account',
      commands: ['gsutil iam ch user:<email>:objectAdmin gs://<bucket>'],
    },
    {
      id: 'dataset-lineage',
      name: 'Track Dataset Lineage',
      problem: 'Need to know which datasets were used for training runs',
      solution: 'Use data register with --parent for lineage and --mlflow-run for training link',
      commands: [
        'akk data download - Download and auto-register as raw:1',
        'akk data register ./augmented.csv --name aug --parent raw:1 - Track derivation',
        'akk data register ./train.db --name train --mlflow-run <id> - Link to training',
        'akk data list --mlflow - View all datasets with MLflow links',
      ],
    },
    {
      id: 'transformers-deprecations',
      name: 'Fix Transformers Deprecations',
      problem: 'Training fails with TypeError on evaluation_strategy or as_target_tokenizer',
      solution:
        'Use modern API: eval_strategy instead of evaluation_strategy, tokenizer(text, text_target=target) instead of as_target_tokenizer()',
      commands: [
        'akk preflight check train.py - Detects deprecated APIs',
        'akk notebook build training.toml - Generates notebooks with modern APIs',
      ],
    },
    {
      id: 'disk-space-kaggle',
      name: 'Fix Kaggle Disk Space Errors',
      problem: 'Notebook fails with "disk space exceeded" on Kaggle',
      solution:
        'Kaggle has ~10GB effective disk. Checkpoints are fp32 (2x model size). Set save_total_limit=1, clear HF cache after model load.',
      commands: [
        'akk preflight check train.py --platform kaggle-p100 --verbose - Check peak disk usage',
        'akk notebook build training.toml - Auto-configures disk optimizations',
      ],
    },
    {
      id: 'training-toml-config',
      name: 'Training TOML Configuration',
      problem: 'Need reproducible, validated notebook generation',
      solution: 'Use training.toml as single source of truth for all ML decisions',
      config: `[meta]
name = "byt5-akkadian"
version = "1.0.11"  # Drives model name: akkadian-byt5-v1-0-11

[model]
name = "google/byt5-base"
src_lang = "akk"
tgt_lang = "eng"

[training]
num_epochs = 8
batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4

[kaggle_model]
handle = "username/model-name"
framework = "transformers"
subtitle = "Brief model description (20-80 chars)"
description = "Full model card in markdown..."
provenance = "Data sources and attribution"
base_model_url = "https://huggingface.co/google/byt5-base"

[submission]
enabled = false  # true for inference notebooks`,
      commands: ['akk notebook build training.toml - Generate notebook from config'],
    },
    {
      id: 'kaggle-model-metadata',
      name: 'Maximize Kaggle Usability Score',
      problem: 'Model uploads have low usability score on Kaggle',
      solution: 'Add rich metadata in training.toml [kaggle_model] section',
      metadata: {
        subtitle: 'Brief tagline (20-80 chars) - REQUIRED for usability score',
        description: 'Full model card (markdown) with Summary, Characteristics, Training Data, Evaluation - REQUIRED',
        provenance: 'Data sources, licenses, attribution - improves score',
        overview: 'Brief instance summary - improves score',
        usage: 'Code examples showing how to load and use the model - HIGHLY RECOMMENDED',
        training_data: 'Array of training data sources - improves score',
        base_model_url: 'Link to HuggingFace base model - improves score',
        fine_tunable: 'Whether model can be fine-tuned (true/false) - improves score',
      },
      commands: [
        'akk notebook build training.toml - Generates both model-metadata.json and model-instance-metadata.json',
      ],
      ai_agent_guidance: {
        when_uploading_models: [
          'ALWAYS include subtitle (20-80 chars) - this is the most visible field',
          'ALWAYS include description with markdown model card',
          'ALWAYS include usage examples showing how to load the model',
          'Include provenance for data attribution and licensing',
          'Set fine_tunable: true if the model can be further fine-tuned',
        ],
        description_template: `## Summary
Brief 1-2 sentence description of what the model does.

## Model Details
- **Base model**: [link to HuggingFace model]
- **Task**: Translation / Text generation / etc.
- **Languages**: Source → Target
- **Parameters**: XXM

## Training Data
- Dataset name and size
- Any data preprocessing applied

## Evaluation
- Metrics achieved (BLEU, chrF, etc.)
- Validation loss

## Usage
\`\`\`python
import kagglehub
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = kagglehub.model_download("owner/model/framework/version")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
\`\`\`

## Limitations
Any known limitations or biases.`,
        example_toml: `[kaggle_model]
handle = "username/akkadian-byt5"
framework = "transformers"
subtitle = "ByT5 fine-tuned for Akkadian to English translation"
description = """
## Summary
ByT5-small fine-tuned on 25K Akkadian-English parallel sentences.

## Training Data
- MTM24 Akkadian Transliterated (19.6K pairs)
- ORACC Combined (5.7K pairs)

## Metrics
- Validation Loss: 0.95
- Training: 12 epochs on Kaggle P100
"""
provenance = "MTM24 competition data (CC-BY), ORACC corpus (CC-BY-SA)"
base_model_url = "https://huggingface.co/google/byt5-small"
fine_tunable = true`,
      },
    },
    {
      id: 'internet-submission-mismatch',
      name: 'Fix Internet/Submission Mismatch',
      problem: 'Competition submission fails or kernel runs but no score appears',
      solution: 'Kaggle competition kernels MUST have internet=false. Use two-stage workflow.',
      checks: [
        'Preflight detects submission.csv output with internet=true',
        'Preflight detects HuggingFace from_pretrained() with internet=false',
        'Use training notebook (internet ON) + inference notebook (internet OFF)',
      ],
      commands: ['akk preflight check train.py - Catches internet/submission mismatches'],
    },
  ],

  errors: [
    {
      code: 'OOM',
      message: 'CUDA out of memory',
      cause: 'Model + batch too large for GPU VRAM',
      fix: 'Reduce batch_size, use gradient_accumulation_steps, enable fp16',
    },
    {
      code: 'DISK_SPACE',
      message: 'Your notebook tried to use more disk space than is available',
      cause:
        'Kaggle has ~10GB effective disk. Checkpoints are fp32 (2x model size). Multiple checkpoints during rotation.',
      fix: 'Set save_total_limit=1, clear HF cache after model load, use akk notebook build for auto-optimization',
    },
    {
      code: 'EVALUATION_STRATEGY',
      message: 'TypeError: Seq2SeqTrainingArguments.__init__() got an unexpected keyword argument evaluation_strategy',
      cause: 'evaluation_strategy is deprecated in transformers>=4.46',
      fix: 'Use eval_strategy instead. Run akk preflight check to detect, or use akk notebook build for modern APIs.',
    },
    {
      code: 'AS_TARGET_TOKENIZER',
      message: 'as_target_tokenizer() is deprecated',
      cause: 'Old tokenization API deprecated in transformers>=4.40',
      fix: 'Use tokenizer(text, text_target=target) instead',
    },
    {
      code: 'QUOTA_EXCEEDED',
      message: 'Kaggle API quota exceeded',
      cause: 'Too many API calls in 24h period',
      fix: 'Wait 24h or use different account',
    },
    {
      code: 'NO_COMPETITION_CONFIG',
      message: 'No competition.toml found',
      cause: 'Not in a competition directory',
      fix: 'Run: akk competition init <slug>',
    },
    {
      code: 'HUGGINGFACE_NO_INTERNET',
      message: "OSError: We couldn't connect to 'https://huggingface.co'",
      cause: 'Notebook uses from_pretrained() with HuggingFace model but internet is disabled',
      fix: 'Use two-stage workflow: Training notebook (internet ON) uploads to Kaggle registry, Inference notebook (internet OFF) loads via kagglehub.model_download()',
    },
    {
      code: 'SUBMISSION_INTERNET_ON',
      message: 'Kernel completed but no competition score',
      cause: 'Notebook outputs submission.csv but has internet=true. Competition submissions require internet=false.',
      fix: 'Set enable_internet=false in kernel-metadata.json, or use akk notebook build with submission.enabled=true',
    },
    {
      code: 'PIP_INSTALL_NO_INTERNET',
      message: 'ERROR: No matching distribution found for <package>',
      cause: 'pip install fails because internet is disabled',
      fix: 'Use only Kaggle pre-installed packages when internet=false. sacrebleu/evaluate are NOT pre-installed.',
    },
    {
      code: 'SACREBLEU_NOT_FOUND',
      message: "ModuleNotFoundError: No module named 'sacrebleu'",
      cause: 'sacrebleu is not pre-installed on Kaggle',
      fix: 'For offline inference, use eval_loss instead of custom metrics. akk notebook build handles this automatically.',
    },
  ],

  config: [
    {
      file: 'competition.toml',
      description: 'Competition-specific configuration',
      schema: {
        'competition.name': { type: 'string', description: 'Competition name' },
        'competition.slug': { type: 'string', description: 'Competition slug' },
        'competition.kaggle.username': {
          type: 'string',
          description: 'Kaggle username',
        },
        'competition.kaggle.kernel_versioning.strategy': {
          type: 'enum',
          description: 'semver | timestamp | experiment | overwrite',
          default: 'semver',
        },
        'training.default_platform': {
          type: 'string',
          description: 'Default platform',
          default: 'kaggle-p100',
        },
        'training.default_batch_size': {
          type: 'number',
          description: 'Batch size',
          default: '2',
        },
      },
    },
    {
      file: 'akk.toml',
      description: 'Global CLI configuration',
      schema: {
        'kaggle.username': { type: 'string', description: 'Kaggle username' },
        'colab.gcs_bucket': {
          type: 'string',
          description: 'GCS bucket for Colab',
        },
        'mlflow.port': {
          type: 'number',
          description: 'MLflow server port',
          default: '5001',
        },
      },
    },
    {
      file: 'training.toml',
      description: 'Training notebook configuration (single source of truth for ML decisions)',
      schema: {
        'meta.name': { type: 'string', description: 'Notebook/model name' },
        'meta.version': {
          type: 'string',
          description: 'Semver version (drives model naming: 1.0.11 → akkadian-byt5-v1-0-11)',
        },
        'model.name': {
          type: 'string',
          description: 'HuggingFace model name (e.g., google/byt5-base)',
        },
        'training.num_epochs': {
          type: 'number',
          description: 'Training epochs',
          default: '8',
        },
        'training.batch_size': {
          type: 'number',
          description: 'Batch size',
          default: '2',
        },
        'training.gradient_accumulation_steps': {
          type: 'number',
          description: 'Gradient accumulation',
          default: '4',
        },
        'training.learning_rate': {
          type: 'number',
          description: 'Learning rate',
          default: '2e-4',
        },
        'kaggle_model.handle': {
          type: 'string',
          description: 'Kaggle model handle (e.g., username/model-name)',
        },
        'kaggle_model.subtitle': {
          type: 'string',
          description: 'Model subtitle (20-80 chars for usability score)',
        },
        'kaggle_model.description': {
          type: 'string',
          description: 'Full model card (markdown)',
        },
        'kaggle_model.provenance': {
          type: 'string',
          description: 'Data provenance/attribution',
        },
        'kaggle_model.base_model_url': {
          type: 'string',
          description: 'Link to HuggingFace base model',
        },
        'submission.enabled': {
          type: 'boolean',
          description: 'Enable submission.csv generation (sets internet=false)',
          default: 'false',
        },
      },
    },
  ],

  platforms: {
    'kaggle-p100': {
      name: 'Kaggle P100',
      limits: {
        gpu_memory: '16 GB',
        ram: '13 GB',
        disk: '10 GB effective (HF cache + /kaggle/working share partition)',
        runtime: '9 hours',
      },
      tips: [
        'Use batch_size=2 with gradient_accumulation_steps=8 for 600M models',
        'Enable fp16 for memory efficiency',
        'Set save_total_limit=1 to minimize disk usage',
        'Clear HF cache after model load to free ~3GB',
        'Checkpoints save in fp32 (2x model size) - budget accordingly',
      ],
    },
    'kaggle-t4x2': {
      name: 'Kaggle T4 x2',
      limits: {
        gpu_memory: '15 GB x2',
        ram: '13 GB',
        disk: '10 GB effective',
        runtime: '9 hours',
      },
      tips: [
        'Use accelerate for multi-GPU training',
        'Effective batch size doubles with 2 GPUs',
        'Set save_total_limit=1 to minimize disk usage',
      ],
    },
    'colab-pro': {
      name: 'Colab Pro',
      limits: {
        gpu_memory: '40 GB (A100) or 16 GB (V100)',
        ram: '52 GB',
        disk: '225 GB',
        runtime: '24 hours',
      },
      tips: [
        'A100 has faster training but may disconnect',
        'Save to GCS frequently for long runs',
        'Use ColabTracker for MLflow integration',
      ],
    },
  },
}
