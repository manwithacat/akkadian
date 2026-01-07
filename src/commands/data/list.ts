/**
 * List registered dataset versions
 */

import { join } from 'path'
import { z } from 'zod'
import { DatasetRegistry } from '../../lib/data-registry'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const ListArgs = z.object({
  name: z.string().optional().describe('Filter by dataset name'),
  source: z.enum(['all', 'kaggle', 'etl', 'derived']).default('all').describe('Filter by source type'),
  mlflow: z.boolean().default(false).describe('Show MLflow linkage'),
  verbose: z.boolean().default(false).describe('Show full metadata'),
})

export const list: CommandDefinition<typeof ListArgs> = {
  name: 'data list',
  description: 'List registered dataset versions',
  help: `
Lists all registered dataset versions from the local registry.

Options:
  --name      Filter by dataset name
  --source    Filter by source type: all, kaggle, etl, derived
  --mlflow    Show MLflow run linkages
  --verbose   Show full metadata including checksums and paths

Output columns:
  NAME     - Dataset name
  VERSION  - Version number
  SOURCE   - Source type (kaggle, etl, derived)
  ROWS     - Number of rows
  SIZE     - File size
  CREATED  - Creation timestamp
  MLFLOW   - Linked MLflow runs (with --mlflow)
`,
  examples: ['akk data list', 'akk data list --name raw', 'akk data list --source kaggle', 'akk data list --mlflow'],
  args: ListArgs,

  async run(args, ctx) {
    const { name, source, mlflow, verbose } = args

    // Get registry
    const dataDir = join(ctx.cwd, 'datasets')
    const registryPath = join(dataDir, 'registry.db')

    // Check if registry exists
    const registryFile = Bun.file(registryPath)
    if (!(await registryFile.exists())) {
      return error('NO_REGISTRY', 'No dataset registry found', 'Run "akk data download" first to create the registry', {
        path: registryPath,
      })
    }

    const registry = new DatasetRegistry(registryPath)

    try {
      // Build filter
      const filter: { name?: string; sourceType?: 'kaggle' | 'etl' | 'derived' } = {}
      if (name) filter.name = name
      if (source !== 'all') {
        filter.sourceType = source as 'kaggle' | 'etl' | 'derived'
      }

      // Get datasets
      const datasets = registry.list(filter)

      if (datasets.length === 0) {
        return success({
          count: 0,
          datasets: [],
          message: filter.name || filter.sourceType ? 'No datasets match the filter' : 'No datasets registered yet',
        })
      }

      // Format output
      const formatted = datasets.map((ds) => {
        const base: Record<string, unknown> = {
          name: ds.name,
          version: ds.version,
          source: ds.sourceType,
          rows: ds.rowCount || null,
          size: ds.sizeBytes ? formatSize(ds.sizeBytes) : null,
          created: ds.createdAt.split('T')[0], // Just date
        }

        if (mlflow) {
          const links = registry.getMlflowLinks(ds.id)
          base.mlflowRuns = links.map((l) => ({
            runId: l.mlflowRunId,
            type: l.linkType,
          }))
        }

        if (verbose) {
          base.id = ds.id
          base.sqlitePath = ds.sqlitePath
          base.checksum = ds.checksum
          base.parentVersionId = ds.parentVersionId
          base.etlPipeline = ds.etlPipeline
          base.metadata = ds.metadata
        }

        return base
      })

      // Group by name for summary
      const names = [...new Set(datasets.map((d) => d.name))]

      return success({
        count: datasets.length,
        names: names.length,
        datasets: formatted,
      })
    } finally {
      registry.close()
    }
  },
}

/**
 * Format byte size to human readable
 */
function formatSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`
}
