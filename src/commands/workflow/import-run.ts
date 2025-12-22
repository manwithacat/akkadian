/**
 * Import a completed GCS run into local MLFlow
 */

import { z } from 'zod'
import { join } from 'path'
import { existsSync, mkdirSync } from 'fs'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { download, listFiles } from '../../lib/gcs'

const ImportRunArgs = z.object({
  run: z.string().describe('Run name to import'),
  experiment: z.string().default('nllb-akkadian').describe('Experiment name'),
  bucket: z.string().optional().describe('GCS bucket'),
  'artifacts-dir': z.string().optional().describe('Local artifacts directory'),
  'skip-model': z.boolean().default(false).describe('Skip downloading model files'),
  register: z.boolean().default(false).describe('Register model in MLFlow'),
  'model-name': z.string().optional().describe('Registered model name'),
})

export const importRun: CommandDefinition<typeof ImportRunArgs> = {
  name: 'workflow import-run',
  description: 'Import a completed GCS run into local MLFlow',
  help: `
Imports a training run from GCS into local MLFlow:

1. Downloads status.json and metrics
2. Downloads model artifacts (unless --skip-model)
3. Imports into MLFlow with proper metrics/params
4. Optionally registers the model

Examples:
  akk workflow import-run --run nllb-v5-original-20251220
  akk workflow import-run --run nllb-v5 --register --model-name nllb-akkadian
`,
  examples: [
    'akk workflow import-run --run nllb-v5-original-20251220',
    'akk workflow import-run --run nllb-v5 --register',
  ],
  args: ImportRunArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket || 'akkadian-byt5-train'
    const experiment = args.experiment
    const runName = args.run

    const gcsPrefix = `gs://${bucket}/mlflow/runs/${experiment}/${runName}`
    const artifactsDir = args['artifacts-dir'] || join(ctx.cwd, 'artifacts', runName)

    // Create artifacts directory
    if (!existsSync(artifactsDir)) {
      mkdirSync(artifactsDir, { recursive: true })
    }

    progress({ step: 'download', message: 'Downloading run metadata...' }, ctx.output)

    // Download status.json
    const statusPath = join(artifactsDir, 'status.json')
    const statusResult = await download(`${gcsPrefix}/status.json`, statusPath)

    if (!statusResult.success) {
      return error('STATUS_NOT_FOUND', `Run not found or incomplete: ${runName}`,
        `Check: gsutil ls ${gcsPrefix}/`, {})
    }

    // Read status
    const statusFile = Bun.file(statusPath)
    let status: any
    try {
      status = JSON.parse(await statusFile.text())
    } catch {
      return error('INVALID_STATUS', 'Could not parse status.json', 'Check file format', {})
    }

    if (status.phase !== 'completed') {
      return error('NOT_COMPLETED', `Run not completed (phase: ${status.phase})`,
        'Wait for training to complete', { status })
    }

    // Download metrics.json if exists
    const metricsPath = join(artifactsDir, 'metrics.json')
    await download(`${gcsPrefix}/metrics.json`, metricsPath)

    // Download sample predictions if exists
    const predictionsPath = join(artifactsDir, 'sample_predictions.json')
    await download(`${gcsPrefix}/artifacts/sample_predictions.json`, predictionsPath)

    // Download model if not skipped
    let modelPath = ''
    if (!args['skip-model']) {
      progress({ step: 'model', message: 'Downloading model files...' }, ctx.output)
      modelPath = join(artifactsDir, 'model')
      mkdirSync(modelPath, { recursive: true })

      // List and download model files
      const modelFiles = await listFiles(`${gcsPrefix}/output/model/`)
      for (const file of modelFiles) {
        const filename = file.split('/').pop() || ''
        if (filename && !filename.startsWith('.')) {
          await download(file, join(modelPath, filename))
        }
      }
    }

    // Import to MLFlow
    progress({ step: 'mlflow', message: 'Importing to MLFlow...' }, ctx.output)

    const mlflowScript = join(ctx.cwd, 'mlflow/scripts/import_run_simple.py')
    const mlflowArgs = [
      'python3', mlflowScript,
      '--run-name', runName,
      '--experiment', experiment,
      '--model-path', modelPath || artifactsDir,
      '--status-json', statusPath,
    ]

    if (existsSync(join(artifactsDir, 'sample_predictions.json'))) {
      mlflowArgs.push('--artifacts-path', artifactsDir)
    }

    if (args.register) {
      mlflowArgs.push('--register-as', args['model-name'] || 'nllb-akkadian')
    }

    const mlflowProc = Bun.spawn(mlflowArgs, {
      stdout: 'inherit',
      stderr: 'inherit',
      env: { ...process.env, MLFLOW_TRACKING_URI: 'http://localhost:5001' },
    })
    await mlflowProc.exited

    if (mlflowProc.exitCode !== 0) {
      return error('MLFLOW_IMPORT_FAILED', 'MLFlow import failed',
        'Check MLFlow server is running: akk mlflow start', {})
    }

    // Calculate geometric mean for Kaggle comparison
    const metrics = status.metrics || {}
    const bleu = metrics.bleu || 0
    const chrf = metrics['chrf++'] || metrics.chrf_pp || metrics.chrf || 0
    const kaggleScore = Math.sqrt(bleu * chrf)

    return success({
      run: runName,
      experiment,
      metrics: {
        bleu: bleu.toFixed(2),
        'chrf++': chrf.toFixed(2),
        'kaggle_score': kaggleScore.toFixed(2),
        duration: status.duration_seconds
          ? `${(status.duration_seconds / 60).toFixed(1)} min`
          : 'unknown',
      },
      artifacts: artifactsDir,
      registered: args.register,
      mlflow_url: `http://localhost:5001/#/experiments`,
    })
  },
}
