/**
 * List kernel versions from registry
 */

import { z } from 'zod'
import { getKernelStatuses, listKernels as listKaggleKernels } from '../../lib/kaggle'
import { getKernelHistory, listRegisteredKernels } from '../../lib/kernel-registry'
import { checkServer, listKernelRuns } from '../../lib/mlflow'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const ListKernelsArgs = z.object({
  name: z.string().optional().describe('Filter by kernel base name'),
  mlflow: z.boolean().default(false).describe('Show MLflow runs instead of local registry'),
  live: z.boolean().default(false).describe('Fetch live status from Kaggle API'),
  user: z.string().optional().describe('Kaggle username for --live mode'),
  json: z.boolean().default(false).describe('Output as JSON'),
})

export const listKernels: CommandDefinition<typeof ListKernelsArgs> = {
  name: 'kaggle list-kernels',
  description: 'List kernel versions from local registry or MLflow',
  help: `
Lists all registered kernel versions, showing version history
and associated metadata tracked in competition.toml.

Use --mlflow to show kernel runs tracked in MLflow instead.
Use --live to fetch real-time status from Kaggle API.

Options:
  --name     Filter by kernel base name
  --mlflow   Show MLflow runs instead of local registry
  --live     Fetch live status from Kaggle API
  --user     Kaggle username (required for --live, defaults to config)
`,
  examples: [
    'akk kaggle list-kernels',
    'akk kaggle list-kernels --name nllb-train',
    'akk kaggle list-kernels --mlflow',
    'akk kaggle list-kernels --live',
  ],
  args: ListKernelsArgs,

  async run(args, ctx) {
    const { name, mlflow: showMlflow, live, user } = args

    // Live mode: fetch status directly from Kaggle API
    if (live) {
      const kaggleUser = user || ctx.config?.kaggle?.username

      if (!kaggleUser) {
        return error(
          'NO_USER',
          'Kaggle username not specified',
          'Provide --user or set kaggle.username in akk.toml',
          {}
        )
      }

      try {
        // Get recent kernels from Kaggle
        const kernelSlugs = await listKaggleKernels(kaggleUser)

        if (kernelSlugs.length === 0) {
          return success({
            source: 'kaggle-live',
            kernels: [],
            message: 'No kernels found',
          })
        }

        // Filter by name if provided
        const filtered = name
          ? kernelSlugs.filter((slug) => slug.toLowerCase().includes(name.toLowerCase()))
          : kernelSlugs.slice(0, 10) // Limit to 10 for performance

        // Fetch live status for each kernel
        const statuses = await getKernelStatuses(filtered)

        return success({
          source: 'kaggle-live',
          kernels: statuses.map((k) => ({
            slug: k.slug,
            status: k.status,
            failureMessage: k.failureMessage || null,
            url: `https://www.kaggle.com/code/${k.slug}`,
          })),
          count: statuses.length,
          running: statuses.filter((k) => k.status === 'running' || k.status === 'queued').length,
        })
      } catch (err) {
        return error(
          'LIVE_STATUS_ERROR',
          err instanceof Error ? err.message : String(err),
          'Check your Kaggle credentials',
          { user: kaggleUser }
        )
      }
    }

    if (showMlflow) {
      // Show MLflow kernel runs
      const mlflowPort = ctx.config?.mlflow?.port || 5001

      if (!(await checkServer(mlflowPort))) {
        return error(
          'MLFLOW_NOT_RUNNING',
          'MLflow server is not running',
          `Start with: akk mlflow start --port ${mlflowPort}`,
          { port: mlflowPort }
        )
      }

      const runs = await listKernelRuns(mlflowPort, { baseName: name })

      if (runs.length === 0) {
        return success({
          source: 'mlflow',
          kernels: [],
          message: name ? `No kernel runs found for "${name}"` : 'No kernel runs found',
        })
      }

      return success({
        source: 'mlflow',
        kernels: runs.map((run) => ({
          runId: run.runId,
          runName: run.runName,
          kernelId: run.kernelId,
          version: run.version,
          status: run.status,
          timestamp: run.timestamp,
          model: run.model,
        })),
        count: runs.length,
      })
    }

    // Show local registry
    if (name) {
      // Show history for specific kernel
      const history = await getKernelHistory(name, ctx.cwd)

      if (history.length === 0) {
        return success({
          source: 'registry',
          kernel: name,
          versions: [],
          message: `No versions found for kernel "${name}"`,
        })
      }

      return success({
        source: 'registry',
        kernel: name,
        versions: history.map((v) => ({
          version: v.version,
          kaggleSlug: v.kaggle_slug,
          timestamp: v.timestamp,
          status: v.status,
          mlflowRunId: v.mlflow_run_id,
          model: v.model,
          notes: v.notes,
        })),
        count: history.length,
      })
    }

    // List all kernels
    const kernels = await listRegisteredKernels(ctx.cwd)

    if (kernels.length === 0) {
      return success({
        source: 'registry',
        kernels: [],
        message: 'No kernels registered. Upload a notebook to create versions.',
      })
    }

    return success({
      source: 'registry',
      kernels: kernels.map((k) => ({
        name: k.name,
        currentVersion: k.currentVersion,
        strategy: k.strategy,
        lastRun: k.lastRun,
        lastStatus: k.lastStatus,
      })),
      count: kernels.length,
    })
  },
}
