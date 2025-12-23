/**
 * Log experiment metrics and parameters to MLFlow
 */

import { z } from 'zod'
import { checkServer, createRun, getOrCreateExperiment, logMetrics, logParams, setRunStatus } from '../../lib/mlflow'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const LogArgs = z.object({
  experiment: z.string().default('akkadian').describe('Experiment name'),
  run: z.string().optional().describe('Run name'),
  params: z.string().optional().describe('Parameters as JSON'),
  metrics: z.string().optional().describe('Metrics as JSON'),
  status: z.enum(['running', 'finished', 'failed']).default('finished').describe('Run status'),
  port: z.number().default(5000).describe('MLFlow server port'),
})

export const log: CommandDefinition<typeof LogArgs> = {
  name: 'mlflow log',
  description: 'Log experiment metrics and parameters',
  help: `
Logs parameters and metrics to an MLFlow experiment.
Creates the experiment and run if they don't exist.

Options:
  --experiment  Experiment name (default: akkadian)
  --run         Run name (default: auto-generated)
  --params      Parameters as JSON string
  --metrics     Metrics as JSON string
  --status      Run status: running, finished, failed (default: finished)
  --port        MLFlow server port (default: 5000)
`,
  examples: [
    'akk mlflow log --metrics \'{"bleu": 23.1, "loss": 0.5}\'',
    'akk mlflow log --experiment nllb --run v1 --params \'{"model": "nllb-600M", "epochs": 3}\'',
    'akk mlflow log --experiment nllb --run v1 --metrics \'{"bleu": 23.1}\' --params \'{"lr": 0.0001}\'',
  ],
  args: LogArgs,

  async run(args, ctx) {
    const { experiment, run: runName, params, metrics, status, port } = args

    // Check server running
    logStep({ step: 'check', message: 'Checking MLFlow server...' }, ctx.output)
    const running = await checkServer(port)

    if (!running) {
      return error('SERVER_NOT_RUNNING', `MLFlow server not running on port ${port}`, 'Start with: akk mlflow start', {
        port,
      })
    }

    // Parse JSON inputs
    let parsedParams: Record<string, string> = {}
    let parsedMetrics: Record<string, number> = {}

    if (params) {
      try {
        parsedParams = JSON.parse(params)
      } catch (e) {
        return error('INVALID_PARAMS', 'Invalid JSON for params', 'Use valid JSON: {"key": "value"}', { params })
      }
    }

    if (metrics) {
      try {
        parsedMetrics = JSON.parse(metrics)
      } catch (e) {
        return error('INVALID_METRICS', 'Invalid JSON for metrics', 'Use valid JSON: {"key": 123.4}', { metrics })
      }
    }

    if (!params && !metrics) {
      return error('NO_DATA', 'No params or metrics provided', 'Provide --params or --metrics', {})
    }

    // Get or create experiment
    logStep({ step: 'experiment', message: `Getting experiment: ${experiment}...` }, ctx.output)
    const experimentId = await getOrCreateExperiment(port, experiment)

    if (!experimentId) {
      return error('EXPERIMENT_FAILED', `Failed to get/create experiment: ${experiment}`, 'Check MLFlow server', {
        experiment,
      })
    }

    // Create run
    logStep({ step: 'run', message: `Creating run${runName ? `: ${runName}` : ''}...` }, ctx.output)
    const runInfo = await createRun(port, experimentId, runName)

    if (!runInfo) {
      return error('RUN_FAILED', 'Failed to create run', 'Check MLFlow server', { experimentId })
    }

    // Log params
    if (Object.keys(parsedParams).length > 0) {
      logStep({ step: 'params', message: 'Logging parameters...' }, ctx.output)
      const paramsSuccess = await logParams(port, runInfo.runId, parsedParams)

      if (!paramsSuccess) {
        return error('PARAMS_FAILED', 'Failed to log parameters', 'Check MLFlow server', { params: parsedParams })
      }
    }

    // Log metrics
    if (Object.keys(parsedMetrics).length > 0) {
      logStep({ step: 'metrics', message: 'Logging metrics...' }, ctx.output)
      const metricsSuccess = await logMetrics(port, runInfo.runId, parsedMetrics)

      if (!metricsSuccess) {
        return error('METRICS_FAILED', 'Failed to log metrics', 'Check MLFlow server', { metrics: parsedMetrics })
      }
    }

    // Set status
    const statusMap = {
      running: 'RUNNING' as const,
      finished: 'FINISHED' as const,
      failed: 'FAILED' as const,
    }

    await setRunStatus(port, runInfo.runId, statusMap[status])

    return success({
      experiment,
      experimentId,
      runId: runInfo.runId,
      runName,
      params: parsedParams,
      metrics: parsedMetrics,
      status,
      url: `http://localhost:${port}/#/experiments/${experimentId}/runs/${runInfo.runId}`,
    })
  },
}
