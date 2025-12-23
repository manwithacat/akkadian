/**
 * Register a dataset version with lineage tracking
 */

import { z } from 'zod'
import { join, basename, dirname, isAbsolute } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { csvToSqlite, computeChecksum, getTableList, getRowCount } from '../../lib/sqlite'
import { DatasetRegistry } from '../../lib/data-registry'
import { parseDatasetRef, type DatasetSourceType } from '../../types/data'
import { Database } from 'bun:sqlite'

const RegisterArgs = z.object({
  path: z.string().describe('Path to CSV or SQLite file to register'),
  name: z.string().describe('Dataset name (e.g., v2_augmented)'),
  parent: z.string().optional().describe('Parent dataset reference (name:version) for lineage'),
  pipeline: z.string().optional().describe('ETL pipeline name that created this'),
  mlflowRun: z.string().optional().describe('MLflow run ID to link'),
  linkType: z.enum(['training', 'evaluation', 'inference']).default('training').describe('MLflow link type'),
  notes: z.string().optional().describe('Additional notes'),
  source: z.enum(['etl', 'derived']).default('etl').describe('Source type'),
})

export const register: CommandDefinition<typeof RegisterArgs> = {
  name: 'data register',
  description: 'Register a dataset version with lineage tracking',
  help: `
Registers a CSV or SQLite file as a tracked dataset version.

This enables:
- Version tracking (name:1, name:2, etc.)
- Lineage tracking (which dataset was derived from which)
- MLflow integration (link datasets to training runs)
- Datasette exploration

Options:
  --name        Dataset name (required)
  --parent      Parent dataset reference for lineage (e.g., raw:1)
  --pipeline    Name of ETL pipeline that created this data
  --mlflow-run  MLflow run ID to link
  --link-type   Type of MLflow link: training, evaluation, inference
  --source      Source type: etl (default) or derived
  --notes       Additional notes stored in metadata

If a CSV file is provided, it will be automatically converted to SQLite.
`,
  examples: [
    'akk data register ./output/augmented.csv --name v2_augmented',
    'akk data register ./output/augmented.csv --name v2_augmented --parent raw:1 --pipeline augmentation_v2',
    'akk data register ./data.db --name processed --mlflow-run abc123',
  ],
  args: RegisterArgs,

  async run(args, ctx) {
    const { path: inputPath, name, parent, pipeline, mlflowRun, linkType, notes, source } = args

    // Resolve path
    const fullPath = isAbsolute(inputPath) ? inputPath : join(ctx.cwd, inputPath)

    // Check file exists
    const inputFile = Bun.file(fullPath)
    if (!(await inputFile.exists())) {
      return error('FILE_NOT_FOUND', `File not found: ${fullPath}`, 'Check the file path', { path: fullPath })
    }

    const dataDir = join(ctx.cwd, 'datasets')
    const registryPath = join(dataDir, 'registry.db')

    // Create datasets dir if needed
    const { mkdir } = await import('fs/promises')
    await mkdir(dataDir, { recursive: true })

    const registry = new DatasetRegistry(registryPath)

    try {
      // Resolve parent if specified
      let parentVersionId: string | undefined
      if (parent) {
        const parentRef = parseDatasetRef(parent)
        const parentDataset = parentRef.version
          ? registry.getVersion(parentRef.name, parentRef.version)
          : registry.getLatestVersion(parentRef.name)

        if (!parentDataset) {
          return error(
            'PARENT_NOT_FOUND',
            `Parent dataset not found: ${parent}`,
            'Check the parent reference or run "akk data list" to see available datasets',
            { parent }
          )
        }

        parentVersionId = parentDataset.id
        progress({ step: 'parent', message: `Linked to parent: ${parentDataset.name}:${parentDataset.version}` }, ctx.output)
      }

      // Determine if conversion is needed
      const ext = basename(fullPath).split('.').pop()?.toLowerCase()
      let sqlitePath: string
      let rowCount = 0

      if (ext === 'csv') {
        // Convert CSV to SQLite
        progress({ step: 'convert', message: 'Converting CSV to SQLite...' }, ctx.output)

        const nextVersion = registry.getNextVersion(name)
        sqlitePath = join(dataDir, `${name}_v${nextVersion}.db`)

        const result = await csvToSqlite({
          inputPath: fullPath,
          outputPath: sqlitePath,
        })

        rowCount = result.totalRows
        progress({ step: 'converted', message: `Created ${sqlitePath} (${rowCount} rows)` }, ctx.output)
      } else if (ext === 'db' || ext === 'sqlite' || ext === 'sqlite3') {
        // Copy SQLite file to datasets dir
        const nextVersion = registry.getNextVersion(name)
        sqlitePath = join(dataDir, `${name}_v${nextVersion}.db`)

        progress({ step: 'copy', message: 'Copying SQLite file...' }, ctx.output)

        const content = await inputFile.arrayBuffer()
        await Bun.write(sqlitePath, content)

        // Count rows
        const db = new Database(sqlitePath, { readonly: true })
        try {
          const tables = getTableList(db)
          for (const table of tables) {
            rowCount += getRowCount(db, table)
          }
        } finally {
          db.close()
        }
      } else {
        return error(
          'UNSUPPORTED_FORMAT',
          `Unsupported file format: .${ext}`,
          'Supported formats: .csv, .db, .sqlite, .sqlite3',
          { path: fullPath }
        )
      }

      // Compute checksum
      progress({ step: 'checksum', message: 'Computing checksum...' }, ctx.output)
      const checksum = await computeChecksum(sqlitePath)

      // Get file size
      const sqliteFile = Bun.file(sqlitePath)
      const sizeBytes = sqliteFile.size

      // Build metadata
      const metadata: Record<string, unknown> = {}
      if (notes) metadata.notes = notes
      if (pipeline) metadata.pipeline = pipeline
      metadata.originalPath = fullPath

      // Register dataset
      progress({ step: 'register', message: 'Registering dataset...' }, ctx.output)

      const dataset = registry.register({
        name,
        sourceType: source as DatasetSourceType,
        sourcePath: fullPath,
        parentVersionId,
        etlPipeline: pipeline,
        sqlitePath,
        rowCount,
        sizeBytes,
        checksum,
        metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      })

      // Link to MLflow if specified
      if (mlflowRun) {
        progress({ step: 'mlflow', message: `Linking to MLflow run ${mlflowRun}...` }, ctx.output)
        registry.linkMlflowRun(dataset.id, mlflowRun, linkType)
      }

      // Show lineage if parent was specified
      let lineage: string | undefined
      if (parentVersionId) {
        const lineageChain = registry.getLineage(dataset.id)
        lineage = lineageChain.map((d) => `${d.name}:${d.version}`).join(' <- ')
      }

      return success({
        id: dataset.id,
        name: dataset.name,
        version: dataset.version,
        sqlitePath,
        rowCount,
        sizeBytes,
        checksum,
        parent: parent || undefined,
        lineage,
        mlflowRun: mlflowRun || undefined,
      })
    } finally {
      registry.close()
    }
  },
}
