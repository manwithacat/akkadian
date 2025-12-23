/**
 * Cleanup GCS artifacts to manage storage costs
 *
 * GCS Data Retention Policy:
 * - KEEP: output/model/, metrics.json, status.json (essential artifacts)
 * - DELETE: hf_cache/, .config/, sample_data/ (can be re-downloaded)
 * - DELETE: checkpoints/ (intermediate, only keep best model)
 * - OPTIONAL: Keep best checkpoint if --keep-best-checkpoint
 */

import { spawn } from 'child_process'
import { z } from 'zod'
import { deleteFile, exists, getSize, listFiles } from '../../lib/gcs'
import { error, logStep, success, warn } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const CleanupArgs = z.object({
  run: z.string().optional().describe('Specific run to clean up'),
  experiment: z.string().optional().describe('Experiment name'),
  bucket: z.string().optional().describe('GCS bucket'),
  dryRun: z.boolean().default(false).describe('Show what would be deleted without deleting'),
  keepCheckpoints: z.boolean().default(false).describe('Keep all checkpoints'),
  all: z.boolean().default(false).describe('Clean up all runs in experiment'),
  olderThan: z.number().optional().describe('Only clean runs older than N days'),
})

// Patterns to delete (not needed after training)
const DELETE_PATTERNS = [
  'hf_cache/', // HuggingFace cache (base model can be re-downloaded)
  '.config/', // GCloud config (not needed)
  'sample_data/', // Colab default files
  '__pycache__/', // Python cache
  '.ipynb_checkpoints/',
  '.local/',
  '.cache/',
  'pip_packages/', // Installed packages
  '.jupyter/',
]

// Patterns to always keep
const KEEP_PATTERNS = ['output/model/', 'metrics.json', 'status.json', 'training_log.json', 'mlflow_metadata.json']

export const cleanup: CommandDefinition<typeof CleanupArgs> = {
  name: 'colab cleanup',
  description: 'Clean up GCS artifacts to manage storage costs',
  help: `
Removes unnecessary files from GCS training runs to reduce storage costs.

## What Gets Deleted
- hf_cache/: HuggingFace model cache (~1-2GB per model)
- .config/: GCloud configuration
- sample_data/: Colab default sample files
- checkpoints/: Intermediate training checkpoints (by default)

## What Is Kept
- output/model/: Final trained model
- metrics.json: Training metrics
- status.json: Run status
- training_log.json: Training history

## Storage Savings
A typical v4 run was 48GB in GCS. After cleanup:
- Model only: ~1.2GB
- With checkpoints: ~7GB
- Savings: 85-97%

## Examples
  akk colab cleanup --run nllb-v4 --dry-run
  akk colab cleanup --run nllb-v4
  akk colab cleanup --all --older-than 7
  akk colab cleanup --run nllb-v4 --keep-checkpoints
`,
  examples: [
    'akk colab cleanup --run nllb-v4 --dry-run',
    'akk colab cleanup --run nllb-v4',
    'akk colab cleanup --all --experiment nllb-akkadian',
    'akk colab cleanup --all --older-than 7',
  ],
  args: CleanupArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket

    if (!bucket) {
      return error('NO_BUCKET', 'No GCS bucket configured', 'Set colab.gcs_bucket in akk.toml', {})
    }

    const experiment = args.experiment || config?.colab?.default_experiment || 'nllb-akkadian'

    // Get list of runs to clean
    let runs: string[] = []

    if (args.run) {
      runs = [args.run]
    } else if (args.all) {
      // List all runs in experiment
      const prefix = `gs://${bucket}/mlflow/runs/${experiment}/`
      const proc = Bun.spawn(['gsutil', 'ls', prefix], {
        stdout: 'pipe',
        stderr: 'pipe',
      })
      const output = await new Response(proc.stdout).text()

      runs = output
        .trim()
        .split('\n')
        .filter((line) => line.length > 0)
        .map((line) => {
          const match = line.match(/\/([^/]+)\/$/)
          return match ? match[1] : null
        })
        .filter((name): name is string => name !== null)

      if (runs.length === 0) {
        return error('NO_RUNS', 'No runs found in experiment', 'Check the experiment name', { experiment })
      }
    } else {
      return error('NO_TARGET', 'Specify --run or --all', 'Use --run <name> or --all', {})
    }

    let totalDeleted = 0
    let totalSaved = 0
    const results: Record<string, { deleted: string[]; kept: string[]; savedBytes: number }> = {}

    for (const run of runs) {
      logStep({ step: 'analyzing', message: `Analyzing run: ${run}...` }, ctx.output)

      const gcsBase = `gs://${bucket}/mlflow/runs/${experiment}/${run}`
      const deleted: string[] = []
      const kept: string[] = []
      let savedBytes = 0

      // Check age if --older-than specified
      if (args.olderThan) {
        try {
          const statusPath = `${gcsBase}/status.json`
          const proc = Bun.spawn(['gsutil', 'cat', statusPath], {
            stdout: 'pipe',
            stderr: 'pipe',
          })
          const statusText = await new Response(proc.stdout).text()
          if (statusText) {
            const status = JSON.parse(statusText)
            if (status.updated_at) {
              const updatedAt = new Date(status.updated_at)
              const age = (Date.now() - updatedAt.getTime()) / (1000 * 60 * 60 * 24)
              if (age < args.olderThan) {
                console.log(`  Skipping ${run} (${age.toFixed(1)} days old, < ${args.olderThan} days)`)
                continue
              }
            }
          }
        } catch {
          // If we can't read logStep, proceed with cleanup
        }
      }

      // List all files in run
      const proc = Bun.spawn(['gsutil', 'ls', '-r', gcsBase + '/'], {
        stdout: 'pipe',
        stderr: 'pipe',
      })
      const output = await new Response(proc.stdout).text()
      const allFiles = output
        .trim()
        .split('\n')
        .filter((line) => line.length > 0 && !line.endsWith(':'))

      // Categorize files
      for (const file of allFiles) {
        const relPath = file.replace(gcsBase + '/', '')

        // Check if should delete
        const shouldDelete =
          DELETE_PATTERNS.some((pattern) => relPath.includes(pattern)) ||
          (!args.keepCheckpoints && relPath.includes('checkpoints/'))

        // Check if should keep
        const mustKeep = KEEP_PATTERNS.some((pattern) => relPath.includes(pattern))

        if (mustKeep) {
          kept.push(relPath)
        } else if (shouldDelete) {
          deleted.push(file)

          // Get size if not dry run
          const sizeProc = Bun.spawn(['gsutil', 'du', '-s', file], {
            stdout: 'pipe',
            stderr: 'pipe',
          })
          const sizeOutput = await new Response(sizeProc.stdout).text()
          const sizeMatch = sizeOutput.match(/^(\d+)/)
          if (sizeMatch) {
            savedBytes += parseInt(sizeMatch[1], 10)
          }
        }
      }

      // Delete files
      if (!args.dryRun && deleted.length > 0) {
        logStep({ step: 'deleting', message: `Deleting ${deleted.length} files...` }, ctx.output)

        for (const file of deleted) {
          const delProc = Bun.spawn(['gsutil', 'rm', file], {
            stdout: 'pipe',
            stderr: 'pipe',
          })
          await delProc.exited
        }

        totalDeleted += deleted.length
      }

      totalSaved += savedBytes
      results[run] = { deleted, kept, savedBytes }

      if (args.dryRun) {
        console.log(`\n${run}:`)
        console.log(`  Would delete: ${deleted.length} files (${formatBytes(savedBytes)})`)
        console.log(`  Would keep: ${kept.length} files`)
        if (deleted.length > 0 && deleted.length <= 20) {
          console.log(`  Files to delete:`)
          for (const file of deleted.slice(0, 10)) {
            console.log(`    - ${file.replace(gcsBase + '/', '')}`)
          }
          if (deleted.length > 10) {
            console.log(`    ... and ${deleted.length - 10} more`)
          }
        }
      }
    }

    if (args.dryRun) {
      return success({
        dryRun: true,
        runs: runs.length,
        totalFiles: Object.values(results).reduce((sum, r) => sum + r.deleted.length, 0),
        estimatedSavings: formatBytes(totalSaved),
        message: 'Run without --dry-run to delete files',
      })
    }

    return success({
      runs: runs.length,
      filesDeleted: totalDeleted,
      bytesSaved: totalSaved,
      savedFormatted: formatBytes(totalSaved),
      results,
      message: `Cleaned up ${totalDeleted} files, saved ${formatBytes(totalSaved)}`,
    })
  },
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / k ** i).toFixed(2)) + ' ' + sizes[i]
}
