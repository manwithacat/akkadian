/**
 * Sync MLFlow runs from GCS
 */

import { z } from 'zod'
import { join } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'

const SyncArgs = z.object({
  bucket: z.string().default('akkadian-models').describe('GCS bucket name'),
  experiment: z.string().optional().describe('Experiment name to sync'),
  run: z.string().optional().describe('Specific run name'),
  list: z.boolean().default(false).describe('List remote experiments/runs'),
  port: z.number().default(5001).describe('MLFlow server port'),
  artifacts: z.boolean().default(false).describe('Download artifacts'),
})

export const sync: CommandDefinition<typeof SyncArgs> = {
  name: 'mlflow sync',
  description: 'Sync MLFlow runs from GCS',
  help: `
Syncs MLFlow experiment runs from GCS to the local tracking server.
Requires the sync_from_gcs.py script in mlflow/scripts/.

Options:
  --bucket      GCS bucket name (default: akkadian-models)
  --experiment  Experiment name to sync
  --run         Specific run name to sync
  --list        List remote experiments/runs
  --port        MLFlow server port (default: 5001)
  --artifacts   Download artifacts (large files)
`,
  examples: [
    'akk mlflow sync --list',
    'akk mlflow sync --experiment nllb-akkadian --list',
    'akk mlflow sync --experiment nllb-akkadian',
    'akk mlflow sync --experiment nllb-akkadian --run nllb-1.3B-v4 --artifacts',
  ],
  args: SyncArgs,

  async run(args, ctx) {
    const { bucket, experiment, run: runName, list, port, artifacts } = args

    // Find sync script
    const syncScript = join(ctx.cwd, 'mlflow', 'scripts', 'sync_from_gcs.py')
    const scriptExists = await Bun.file(syncScript).exists()

    if (!scriptExists) {
      return error(
        'SCRIPT_NOT_FOUND',
        `Sync script not found: ${syncScript}`,
        'Create mlflow/scripts/sync_from_gcs.py',
        { path: syncScript }
      )
    }

    // Build command
    const pythonArgs = ['python3', syncScript, '--bucket', bucket, '--tracking-uri', `http://localhost:${port}`]

    if (list) {
      pythonArgs.push('--list')
      if (experiment) {
        pythonArgs.push('--experiment', experiment)
      }
    } else if (experiment) {
      pythonArgs.push('--experiment', experiment)
      if (runName) {
        pythonArgs.push('--run', runName)
      }
      if (artifacts) {
        pythonArgs.push('--download-artifacts')
      }
    } else if (!list) {
      return error('NO_EXPERIMENT', 'Specify --experiment or use --list', 'Example: akk mlflow sync --experiment nllb-akkadian', {})
    }

    progress({ step: 'sync', message: `Syncing from gs://${bucket}...` }, ctx.output)

    // Run sync script
    const proc = Bun.spawn(pythonArgs, {
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: ctx.cwd,
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error('SYNC_FAILED', `Sync failed: ${stderr || stdout}`, 'Check GCS credentials and bucket access', {
        exitCode,
        stderr,
      })
    }

    // Parse output
    const lines = stdout.trim().split('\n')

    if (list) {
      // Parse list output
      const items = lines.filter((l) => l.startsWith('  - ')).map((l) => l.replace('  - ', ''))

      return success({
        bucket,
        experiment: experiment || null,
        items,
        count: items.length,
      })
    }

    return success({
      bucket,
      experiment,
      run: runName || 'all',
      artifacts,
      log: lines.slice(-10),
      url: `http://localhost:${port}`,
    })
  },
}
