/**
 * List runs in GCS
 */

import { z } from 'zod'
import { download } from '../../lib/gcs'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const ListRunsArgs = z.object({
  experiment: z.string().default('nllb-akkadian').describe('Experiment name'),
  bucket: z.string().optional().describe('GCS bucket'),
  status: z.enum(['all', 'completed', 'running', 'failed']).default('all').describe('Filter by status'),
  limit: z.number().default(20).describe('Max runs to show'),
})

export const listRuns: CommandDefinition<typeof ListRunsArgs> = {
  name: 'workflow list-runs',
  description: 'List training runs in GCS',
  help: `
Lists training runs stored in GCS:

Examples:
  akk workflow list-runs
  akk workflow list-runs --experiment nllb-akkadian --status completed
  akk workflow list-runs --limit 10
`,
  examples: ['akk workflow list-runs', 'akk workflow list-runs --status completed'],
  args: ListRunsArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket || 'akkadian-byt5-train'
    const experiment = args.experiment

    const gcsPrefix = `gs://${bucket}/mlflow/runs/${experiment}/`

    // List run directories
    const proc = Bun.spawn(['gsutil', 'ls', gcsPrefix], {
      stdout: 'pipe',
      stderr: 'pipe',
    })
    await proc.exited

    if (proc.exitCode !== 0) {
      const stderr = await new Response(proc.stderr).text()
      return error('LIST_FAILED', 'Could not list runs', stderr, {})
    }

    const stdout = await new Response(proc.stdout).text()
    const runDirs = stdout
      .trim()
      .split('\n')
      .filter((line) => line.endsWith('/'))
      .map((line) => {
        const parts = line.replace(/\/$/, '').split('/')
        return parts[parts.length - 1]
      })
      .filter((name) => name && name !== experiment)

    // Get status for each run (up to limit)
    const runs: Array<{
      name: string
      phase: string
      bleu?: number
      chrf?: number
      kaggle?: number
      duration?: string
      updated?: string
    }> = []

    for (const runName of runDirs.slice(0, args.limit)) {
      const tempFile = `/tmp/status_${Date.now()}_${runName}.json`
      const statusResult = await download(`${gcsPrefix}${runName}/status.json`, tempFile)

      let phase = 'unknown'
      let bleu: number | undefined
      let chrf: number | undefined
      let duration: string | undefined
      let updated: string | undefined

      if (statusResult.success) {
        try {
          const status = JSON.parse(await Bun.file(tempFile).text())
          phase = status.phase || 'unknown'
          updated = status.updated_at

          if (status.metrics) {
            bleu = status.metrics.bleu
            chrf = status.metrics['chrf++'] || status.metrics.chrf_pp || status.metrics.chrf
          }

          if (status.duration_seconds) {
            duration = `${(status.duration_seconds / 60).toFixed(0)}m`
          }
        } catch {
          // Skip invalid status files
        }
      }

      // Apply status filter
      if (args.status !== 'all') {
        if (args.status === 'completed' && phase !== 'completed') continue
        if (args.status === 'running' && phase !== 'training') continue
        if (args.status === 'failed' && phase !== 'failed') continue
      }

      const kaggle = bleu && chrf ? Math.sqrt(bleu * chrf) : undefined

      runs.push({
        name: runName,
        phase,
        bleu,
        chrf,
        kaggle,
        duration,
        updated,
      })
    }

    // Sort by updated time (most recent first)
    runs.sort((a, b) => {
      if (!a.updated) return 1
      if (!b.updated) return -1
      return b.updated.localeCompare(a.updated)
    })

    // Format output
    console.log(`\nRuns in ${experiment}:\n`)
    console.log(
      'RUN'.padEnd(42) +
        'STATUS'.padEnd(12) +
        'BLEU'.padStart(8) +
        'chrF++'.padStart(8) +
        'KAGGLE'.padStart(8) +
        'TIME'.padStart(8)
    )
    console.log('-'.repeat(86))

    for (const run of runs) {
      const statusIcon =
        run.phase === 'completed'
          ? '\u2713'
          : run.phase === 'training'
            ? '\u2022'
            : run.phase === 'failed'
              ? '\u2717'
              : '?'

      console.log(
        `${statusIcon} ${run.name.slice(0, 38).padEnd(38)} ` +
          `${run.phase.slice(0, 10).padEnd(10)} ` +
          `${run.bleu?.toFixed(1).padStart(7) || '    -  '} ` +
          `${run.chrf?.toFixed(1).padStart(7) || '    -  '} ` +
          `${run.kaggle?.toFixed(1).padStart(7) || '    -  '} ` +
          `${(run.duration || '-').padStart(7)}`
      )
    }

    console.log('')

    return success({
      experiment,
      total: runs.length,
      runs: runs.map((r) => ({
        name: r.name,
        phase: r.phase,
        metrics: r.bleu ? { bleu: r.bleu, chrf: r.chrf, kaggle: r.kaggle } : undefined,
      })),
    })
  },
}
