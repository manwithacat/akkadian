/**
 * Download training artifacts from GCS
 */

import { join } from 'path'
import { z } from 'zod'
import { download, getSize, listFiles, rsync } from '../../lib/gcs'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const DownloadArtifactsArgs = z.object({
  run: z.string().describe('Run name to download'),
  bucket: z.string().optional().describe('GCS bucket'),
  experiment: z.string().optional().describe('Experiment name'),
  output: z.string().optional().describe('Local output directory'),
  model: z.boolean().default(true).describe('Download trained model'),
  checkpoints: z.boolean().default(false).describe('Download all checkpoints'),
  metrics: z.boolean().default(true).describe('Download metrics and logs'),
  all: z.boolean().default(false).describe('Download everything'),
})

export const downloadArtifacts: CommandDefinition<typeof DownloadArtifactsArgs> = {
  name: 'colab download-artifacts',
  description: 'Download training artifacts from GCS',
  help: `
Downloads trained model, checkpoints, metrics, and logs from GCS.

Artifacts are organized as:
  gs://{bucket}/mlflow/runs/{experiment}/{run}/
    ├── output/           # Final trained model
    │   ├── model/
    │   └── tokenizer/
    ├── checkpoints/      # Training checkpoints
    │   ├── checkpoint-100/
    │   └── checkpoint-200/
    ├── metrics.json      # Final metrics
    ├── training_log.json # Training history
    └── status.json       # Run status

Options:
  --run          Run name (required)
  --bucket       GCS bucket (default: from config)
  --experiment   Experiment name (default: 'default')
  --output       Local output directory (default: ./artifacts/{run})
  --model        Download trained model (default: true)
  --checkpoints  Download all checkpoints (default: false)
  --metrics      Download metrics and logs (default: true)
  --all          Download everything
`,
  examples: [
    'akk colab download-artifacts --run nllb-v4',
    'akk colab download-artifacts --run nllb-v4 --all --output ./models/nllb-v4',
    'akk colab download-artifacts --run nllb-v4 --checkpoints',
  ],
  args: DownloadArtifactsArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket

    if (!bucket) {
      return error('NO_BUCKET', 'No GCS bucket configured', 'Set colab.gcs_bucket in akk.toml', {})
    }

    const experiment = args.experiment || 'default'
    const prefix = `mlflow/runs/${experiment}/${args.run}`
    const gcsBase = `gs://${bucket}/${prefix}`

    // Determine output directory
    const outputDir = args.output
      ? args.output.startsWith('/')
        ? args.output
        : join(ctx.cwd, args.output)
      : join(ctx.cwd, 'artifacts', args.run)

    // Create output directory
    await Bun.write(join(outputDir, '.gitkeep'), '')

    const downloaded: string[] = []
    const failed: string[] = []
    let _totalSize = 0

    // Download model
    if (args.model || args.all) {
      logStep({ step: 'model', message: 'Downloading trained model...' }, ctx.output)

      const modelPath = `${gcsBase}/output/`
      const modelFiles = await listFiles(modelPath)

      if (modelFiles.length > 0) {
        const modelSize = (await getSize(modelPath)) || 0
        _totalSize += modelSize

        const modelDir = join(outputDir, 'model')
        const result = await rsync(modelPath, modelDir)

        if (result.success) {
          downloaded.push('model')
        } else {
          failed.push(`model: ${result.message}`)
        }
      } else {
        // Try alternative paths
        const altPaths = [`${gcsBase}/model/`, `${gcsBase}/trained_model/`, `${gcsBase}/final_model/`]

        for (const altPath of altPaths) {
          const files = await listFiles(altPath)
          if (files.length > 0) {
            const modelDir = join(outputDir, 'model')
            const result = await rsync(altPath, modelDir)
            if (result.success) {
              downloaded.push('model')
              break
            }
          }
        }
      }
    }

    // Download checkpoints
    if (args.checkpoints || args.all) {
      logStep({ step: 'checkpoints', message: 'Downloading checkpoints...' }, ctx.output)

      const checkpointPath = `${gcsBase}/checkpoints/`
      const checkpointFiles = await listFiles(checkpointPath)

      if (checkpointFiles.length > 0) {
        const checkpointDir = join(outputDir, 'checkpoints')
        const result = await rsync(checkpointPath, checkpointDir)

        if (result.success) {
          const numCheckpoints = checkpointFiles.filter((f) => f.includes('checkpoint-')).length
          downloaded.push(`checkpoints (${numCheckpoints})`)
        } else {
          failed.push(`checkpoints: ${result.message}`)
        }
      }
    }

    // Download metrics and logs
    if (args.metrics || args.all) {
      logStep({ step: 'metrics', message: 'Downloading metrics and logs...' }, ctx.output)

      const metricFiles = ['metrics.json', 'training_log.json', 'status.json', 'config.json', 'eval_results.json']

      for (const filename of metricFiles) {
        const gcsPath = `${gcsBase}/${filename}`
        const localPath = join(outputDir, filename)
        const result = await download(gcsPath, localPath)

        if (result.success) {
          downloaded.push(filename)
        }
      }

      // Also download MLFlow metadata
      const mlflowPath = `${gcsBase}/mlflow_metadata.json`
      const mlflowLocal = join(outputDir, 'mlflow_metadata.json')
      await download(mlflowPath, mlflowLocal)
    }

    // Download everything else if --all
    if (args.all) {
      logStep({ step: 'artifacts', message: 'Downloading all artifacts...' }, ctx.output)

      const result = await rsync(gcsBase + '/', outputDir, { delete: false })
      if (result.success) {
        downloaded.push('all artifacts')
      }
    }

    if (downloaded.length === 0 && failed.length === 0) {
      return error('NO_ARTIFACTS', `No artifacts found for run: ${args.run}`, 'Check the run name and experiment', {
        gcsPath: gcsBase,
      })
    }

    // Read metrics if available
    let metrics: Record<string, number> | null = null
    try {
      const metricsPath = join(outputDir, 'metrics.json')
      const metricsFile = Bun.file(metricsPath)
      if (await metricsFile.exists()) {
        metrics = JSON.parse(await metricsFile.text())
      }
    } catch {
      // Ignore
    }

    return success({
      run: args.run,
      experiment,
      outputDir,
      downloaded,
      failed: failed.length > 0 ? failed : undefined,
      metrics,
      gcsSource: gcsBase,
      nextSteps: [
        'Sync to MLFlow: akk mlflow sync --experiment ' + experiment,
        'Analyze results: akk local analyze --run ' + args.run,
        'Upload to Kaggle: akk kaggle upload-model ' + outputDir + '/model',
      ],
    })
  },
}
