/**
 * Check Colab training status by monitoring GCS
 */

import { z } from 'zod'
import { download, listFiles } from '../../lib/gcs'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const StatusArgs = z.object({
  run: z.string().optional().describe('Run name to check'),
  bucket: z.string().optional().describe('GCS bucket'),
  experiment: z.string().optional().describe('Experiment name'),
  watch: z.boolean().default(false).describe('Continuously watch for updates'),
  interval: z.number().default(30).describe('Watch interval in seconds'),
})

interface TrainingStatus {
  phase: 'not_started' | 'initializing' | 'training' | 'evaluating' | 'completed' | 'failed'
  progress?: number
  currentEpoch?: number
  totalEpochs?: number
  lastCheckpoint?: string
  metrics?: Record<string, number>
  error?: string
}

async function parseStatusFile(bucket: string, prefix: string): Promise<TrainingStatus | null> {
  const statusPath = `gs://${bucket}/${prefix}/status.json`
  const tempFile = `/tmp/status_${Date.now()}.json`

  const result = await download(statusPath, tempFile)
  if (!result.success) {
    return null
  }

  try {
    const content = await Bun.file(tempFile).text()
    return JSON.parse(content) as TrainingStatus
  } catch {
    return null
  }
}

async function getLatestCheckpoint(bucket: string, prefix: string): Promise<string | null> {
  const checkpointPath = `gs://${bucket}/${prefix}/checkpoints/`
  const files = await listFiles(checkpointPath)

  if (files.length === 0) return null

  // Sort by checkpoint number
  const checkpoints = files
    .filter((f) => f.includes('checkpoint-'))
    .sort((a, b) => {
      const numA = parseInt(a.match(/checkpoint-(\d+)/)?.[1] || '0', 10)
      const numB = parseInt(b.match(/checkpoint-(\d+)/)?.[1] || '0', 10)
      return numB - numA
    })

  return checkpoints[0] || null
}

async function getMetrics(bucket: string, prefix: string): Promise<Record<string, number> | null> {
  const metricsPath = `gs://${bucket}/${prefix}/metrics.json`
  const tempFile = `/tmp/metrics_${Date.now()}.json`

  const result = await download(metricsPath, tempFile)
  if (!result.success) {
    return null
  }

  try {
    const content = await Bun.file(tempFile).text()
    return JSON.parse(content)
  } catch {
    return null
  }
}

export const status: CommandDefinition<typeof StatusArgs> = {
  name: 'colab status',
  description: 'Check Colab training status from GCS',
  help: `
Monitors training progress by checking status files in GCS.

The training notebook should write status updates to:
  gs://{bucket}/mlflow/runs/{experiment}/{run}/status.json

Status file format:
  {
    "phase": "training",
    "currentEpoch": 2,
    "totalEpochs": 5,
    "progress": 0.4,
    "metrics": {"loss": 0.5, "bleu": 15.2}
  }

Options:
  --run         Run name to check
  --bucket      GCS bucket (default: from config)
  --experiment  Experiment name (default: from config or 'default')
  --watch       Continuously watch for updates
  --interval    Watch interval in seconds (default: 30)
`,
  examples: [
    'akk colab status --run nllb-v4',
    'akk colab status --experiment nllb-akkadian --watch',
    'akk colab status --run nllb-v4 --interval 60',
  ],
  args: StatusArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket

    if (!bucket) {
      return error('NO_BUCKET', 'No GCS bucket configured', 'Set colab.gcs_bucket in akk.toml', {})
    }

    const experiment = args.experiment || 'default'

    // If no run specified, list recent runs
    if (!args.run) {
      logStep({ step: 'list', message: 'Listing recent runs...' }, ctx.output)
      const runsPath = `gs://${bucket}/mlflow/runs/${experiment}/`
      const runs = await listFiles(runsPath)

      if (runs.length === 0) {
        return success({
          message: 'No runs found',
          experiment,
          bucket,
        })
      }

      const runNames = runs
        .map((r) => r.replace(runsPath, '').replace('/', ''))
        .filter(Boolean)
        .slice(0, 10)

      return success({
        experiment,
        runs: runNames,
        hint: 'Use --run <name> to check a specific run',
      })
    }

    const prefix = `mlflow/runs/${experiment}/${args.run}`

    const checkStatus = async () => {
      // Try to get status file
      const status = await parseStatusFile(bucket, prefix)
      const checkpoint = await getLatestCheckpoint(bucket, prefix)
      const metrics = await getMetrics(bucket, prefix)

      // Check for completion indicators
      const outputPath = `gs://${bucket}/${prefix}/output/`
      const outputs = await listFiles(outputPath)
      const hasModel = outputs.some((f) => f.includes('model') || f.includes('pytorch'))

      // Infer status if no status file
      const inferredStatus: TrainingStatus = status || {
        phase: hasModel ? 'completed' : checkpoint ? 'training' : 'not_started',
        lastCheckpoint: checkpoint || undefined,
        metrics: metrics || undefined,
      }

      return {
        run: args.run,
        experiment,
        bucket,
        status: inferredStatus,
        outputs: outputs.length,
        hasModel,
        gcsPath: `gs://${bucket}/${prefix}`,
      }
    }

    if (args.watch) {
      console.log(`Watching ${args.run} (Ctrl+C to stop)...`)
      console.log()

      while (true) {
        const result = await checkStatus()
        const now = new Date().toLocaleTimeString()

        console.log(`[${now}] Phase: ${result.status.phase}`)
        if (result.status.currentEpoch !== undefined) {
          console.log(`         Epoch: ${result.status.currentEpoch}/${result.status.totalEpochs}`)
        }
        if (result.status.metrics) {
          const metricsStr = Object.entries(result.status.metrics)
            .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(4) : v}`)
            .join(', ')
          console.log(`         Metrics: ${metricsStr}`)
        }
        if (result.status.lastCheckpoint) {
          console.log(`         Checkpoint: ${result.status.lastCheckpoint}`)
        }

        if (result.status.phase === 'completed' || result.status.phase === 'failed') {
          console.log()
          console.log(`Training ${result.status.phase}!`)
          break
        }

        await new Promise((resolve) => setTimeout(resolve, args.interval * 1000))
      }

      return success(await checkStatus())
    }

    logStep({ step: 'check', message: `Checking status for ${args.run}...` }, ctx.output)
    return success(await checkStatus())
  },
}
