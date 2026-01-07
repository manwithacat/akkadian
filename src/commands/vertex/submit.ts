/**
 * Submit training job to Vertex AI
 */

import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

// Simple info logging
function info(msg: string, _output: any): void {
  console.log(msg)
}

import { spawn } from 'child_process'

const SubmitArgs = z.object({
  // Machine type
  h100: z.boolean().default(false).describe('Use H100 GPU (default: A100-80GB)'),
  a100: z.boolean().default(false).describe('Use A100-80GB GPU'),

  // Training hyperparameters
  epochs: z.number().default(10).describe('Number of training epochs'),
  learningRate: z.number().default(5e-5).describe('Learning rate'),
  weightDecay: z.number().default(0.01).describe('Weight decay'),
  warmupRatio: z.number().default(0.1).describe('Warmup ratio'),
  labelSmoothing: z.number().default(0.0).describe('Label smoothing'),
  batchSize: z.number().default(32).describe('Batch size'),
  gradientAccumulation: z.number().default(2).describe('Gradient accumulation steps'),
  maxLength: z.number().default(256).describe('Max sequence length'),

  // Data
  trainDataset: z.string().default('train_baseline').describe('Training dataset name'),
  validDataset: z.string().default('valid_baseline').describe('Validation dataset name'),
  checkpoint: z.string().optional().describe('GCS path to checkpoint to continue from'),

  // Job config
  jobName: z.string().optional().describe('Custom job name'),
  region: z.string().default('us-central1').describe('GCP region'),

  // Options
  dryRun: z.boolean().default(false).describe('Print command without executing'),
})

export const submit: CommandDefinition<typeof SubmitArgs> = {
  name: 'vertex submit',
  description: 'Submit training job to Vertex AI',
  help: `
Submit a training job to Vertex AI with specified configuration.

Machine types:
  --h100       Use H100 80GB (~$5.50/hr) - fastest
  --a100       Use A100 80GB (~$5.00/hr) - default

Training config:
  --epochs             Number of epochs (default: 10)
  --learning-rate      Learning rate (default: 5e-5)
  --batch-size         Batch size (default: 32)
  --checkpoint         Continue from checkpoint (GCS path)

Data:
  --train-dataset      Training dataset (default: train_baseline)
  --valid-dataset      Validation dataset (default: valid_baseline)
`,
  examples: [
    'akk vertex submit',
    'akk vertex submit --h100 --epochs 15',
    'akk vertex submit --checkpoint gs://akkadian-byt5-train/mlflow/runs/.../output/model',
    'akk vertex submit --learning-rate 3e-5 --batch-size 16',
  ],
  args: SubmitArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = config?.colab?.gcs_bucket || 'akkadian-byt5-train'
    // Use correct project ID (akkadian-481705, not akkadian-ml)
    const project = config?.colab?.project || 'akkadian-481705'

    // Generate job name
    const timestamp = new Date().toISOString().replace(/[:-]/g, '').slice(0, 15)
    const jobName = args.jobName || `nllb-train-${timestamp}`

    // Machine config
    let machineType: string
    let gpuType: string
    if (args.h100) {
      machineType = 'a3-highgpu-1g'
      gpuType = 'NVIDIA_H100_80GB'
    } else {
      machineType = 'a2-ultragpu-1g'
      gpuType = 'NVIDIA_A100_80GB' // Not NVIDIA_TESLA_A100
    }

    info(`Vertex AI Job Submission`, ctx.output)
    info(``, ctx.output)
    info(`Job:      ${jobName}`, ctx.output)
    info(`Machine:  ${machineType} (${gpuType})`, ctx.output)
    info(`Region:   ${args.region}`, ctx.output)
    info(``, ctx.output)
    info(`Training config:`, ctx.output)
    info(`  Epochs:        ${args.epochs}`, ctx.output)
    info(`  Learning rate: ${args.learningRate}`, ctx.output)
    info(`  Batch size:    ${args.batchSize} x ${args.gradientAccumulation}`, ctx.output)
    info(`  Max length:    ${args.maxLength}`, ctx.output)
    if (args.checkpoint) {
      info(`  Checkpoint:    ${args.checkpoint}`, ctx.output)
    }
    info(``, ctx.output)

    // First, upload the training script
    logStep({ step: 'upload', message: 'Uploading training script to GCS...' }, ctx.output)

    const scriptPath = `${process.cwd()}/vertex_ai/trainer/train.py`
    const gcsScriptPath = `gs://${bucket}/vertex/scripts/train.py`

    try {
      await execCommand('gsutil', ['cp', scriptPath, gcsScriptPath])
    } catch (e) {
      return error('UPLOAD_FAILED', 'Failed to upload training script', String(e))
    }

    // Build training arguments
    const trainArgs = [
      `--epochs=${args.epochs}`,
      `--learning_rate=${args.learningRate}`,
      `--weight_decay=${args.weightDecay}`,
      `--warmup_ratio=${args.warmupRatio}`,
      `--label_smoothing=${args.labelSmoothing}`,
      `--batch_size=${args.batchSize}`,
      `--gradient_accumulation=${args.gradientAccumulation}`,
      `--max_length=${args.maxLength}`,
      `--train_dataset=${args.trainDataset}`,
      `--valid_dataset=${args.validDataset}`,
      `--run_name=${jobName}`,
    ]

    if (args.checkpoint) {
      trainArgs.push(`--checkpoint=${args.checkpoint}`)
    }

    // Build the container command
    const containerCommand = [
      'bash',
      '-c',
      `pip install -q transformers>=4.35.0 datasets>=2.14.0 sacrebleu sentencepiece pandas psutil && ` +
        `gsutil cp gs://${bucket}/vertex/scripts/train.py /tmp/train.py && ` +
        `python /tmp/train.py ${trainArgs.join(' ')}`,
    ].join(' ')

    // Build gcloud command
    const gcloudArgs = [
      'ai',
      'custom-jobs',
      'create',
      `--project=${project}`,
      `--region=${args.region}`,
      `--display-name=${jobName}`,
      `--worker-pool-spec=machine-type=${machineType},replica-count=1,accelerator-type=${gpuType},accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest`,
      `--command=${containerCommand}`,
      `--env-vars=GCS_BUCKET=${bucket}`,
    ]

    if (args.dryRun) {
      info(`Dry run - command:`, ctx.output)
      info(`gcloud ${gcloudArgs.join(' ')}`, ctx.output)
      return success({ jobName, dryRun: true })
    }

    // Submit job
    logStep({ step: 'submit', message: 'Submitting job to Vertex AI...' }, ctx.output)

    try {
      const _result = await execCommand('gcloud', gcloudArgs)

      info(``, ctx.output)
      info(`Job submitted successfully!`, ctx.output)
      info(``, ctx.output)
      info(`Monitor at:`, ctx.output)
      info(`  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${project}`, ctx.output)
      info(``, ctx.output)
      info(`Check status:`, ctx.output)
      info(`  akk vertex status --job ${jobName}`, ctx.output)

      return success({
        jobName,
        machine: machineType,
        gpu: gpuType,
        region: args.region,
        output: `gs://${bucket}/vertex/runs/${jobName}`,
      })
    } catch (e) {
      return error('SUBMIT_FAILED', 'Failed to submit job', String(e))
    }
  },
}

async function execCommand(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''

    proc.stdout?.on('data', (data) => {
      stdout += data.toString()
    })
    proc.stderr?.on('data', (data) => {
      stderr += data.toString()
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        reject(new Error(stderr || `Command failed with code ${code}`))
      }
    })
  })
}
