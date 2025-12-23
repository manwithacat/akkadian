/**
 * Download competition data from Kaggle
 */

import { mkdir } from 'fs/promises'
import { join } from 'path'
import { z } from 'zod'
import { DatasetRegistry } from '../../lib/data-registry'
import { error, logStep, success } from '../../lib/output'
import { computeChecksum, csvToSqlite } from '../../lib/sqlite'
import type { CommandDefinition } from '../../types/commands'

const DownloadArgs = z.object({
  competition: z.string().optional().describe('Competition slug (default: from akk.toml)'),
  output: z.string().optional().describe('Output directory (default: datasets)'),
  force: z.boolean().default(false).describe('Overwrite existing files'),
  name: z.string().default('raw').describe('Dataset name for registration'),
  skipSqlite: z.boolean().default(false).describe('Skip SQLite conversion'),
  skipRegister: z.boolean().default(false).describe('Skip dataset registration'),
})

export const download: CommandDefinition<typeof DownloadArgs> = {
  name: 'data download',
  description: 'Download competition data from Kaggle and convert to SQLite',
  help: `
Downloads competition data files from Kaggle and optionally converts them to SQLite
for exploration with Datasette.

By default:
- Downloads to datasets/kaggle/
- Converts CSVs to SQLite
- Registers as a dataset version for lineage tracking

Options:
  --competition   Competition slug (default: from kaggle.competition in akk.toml)
  --output        Output directory (default: datasets)
  --name          Dataset name for registration (default: raw)
  --force         Overwrite existing files
  --skip-sqlite   Don't convert to SQLite
  --skip-register Don't register dataset version
`,
  examples: [
    'akk data download',
    'akk data download --competition deep-past-initiative-machine-translation',
    'akk data download --name competition_v1 --force',
  ],
  args: DownloadArgs,

  async run(args, ctx) {
    const { competition: competitionArg, output, force, name, skipSqlite, skipRegister } = args

    // Get competition from args or config
    const competition = competitionArg || ctx.config?.kaggle?.competition
    if (!competition) {
      return error(
        'NO_COMPETITION',
        'Competition not specified',
        'Use --competition or set kaggle.competition in akk.toml',
        {}
      )
    }

    // Determine paths
    const dataDir = output || join(ctx.cwd, 'datasets')
    const kaggleDir = join(dataDir, 'kaggle')
    const zipPath = join(kaggleDir, `${competition}.zip`)

    // Create directories
    await mkdir(kaggleDir, { recursive: true })

    // Check if already downloaded
    const kaggleFiles = await Bun.file(join(kaggleDir, 'train.csv')).exists()
    if (kaggleFiles && !force) {
      return error('ALREADY_EXISTS', 'Competition data already downloaded', 'Use --force to overwrite', {
        path: kaggleDir,
      })
    }

    // Download from Kaggle
    logStep({ step: 'download', message: `Downloading ${competition}...` }, ctx.output)

    const downloadProc = Bun.spawn(['kaggle', 'competitions', 'download', '-c', competition, '-p', kaggleDir], {
      stdout: 'pipe',
      stderr: 'pipe',
    })

    const downloadStderr = await new Response(downloadProc.stderr).text()
    const downloadExit = await downloadProc.exited

    if (downloadExit !== 0) {
      return error(
        'DOWNLOAD_FAILED',
        `Failed to download: ${downloadStderr}`,
        'Check kaggle credentials and competition slug',
        {
          competition,
        }
      )
    }

    // Unzip if needed
    const zipFile = Bun.file(zipPath)
    if (await zipFile.exists()) {
      logStep({ step: 'extract', message: 'Extracting files...' }, ctx.output)

      const unzipProc = Bun.spawn(['unzip', '-o', zipPath, '-d', kaggleDir], {
        stdout: 'pipe',
        stderr: 'pipe',
      })

      await unzipProc.exited

      // Remove zip file
      await Bun.write(zipPath, '') // Clear file
      const { unlink } = await import('fs/promises')
      await unlink(zipPath)
    }

    // Find CSV files
    const { readdir } = await import('fs/promises')
    const files = await readdir(kaggleDir)
    const csvFiles = files.filter((f) => f.endsWith('.csv'))

    if (csvFiles.length === 0) {
      return error('NO_CSV_FILES', 'No CSV files found in download', 'Check competition data format', {
        path: kaggleDir,
      })
    }

    logStep({ step: 'found', message: `Found ${csvFiles.length} CSV files: ${csvFiles.join(', ')}` }, ctx.output)

    // Convert to SQLite
    let sqlitePath: string | undefined
    let totalRows = 0

    if (!skipSqlite) {
      logStep({ step: 'convert', message: 'Converting to SQLite...' }, ctx.output)

      sqlitePath = join(dataDir, `${name}_v1.db`)

      // Convert each CSV to a table in the same SQLite file
      const { Database } = await import('bun:sqlite')
      const db = new Database(sqlitePath, { create: true })

      try {
        for (const csvFile of csvFiles) {
          const tableName = csvFile.replace(/\.csv$/i, '')
          const csvPath = join(kaggleDir, csvFile)

          logStep({ step: 'table', message: `  Converting ${csvFile} -> ${tableName}...` }, ctx.output)

          const result = await csvToSqlite({
            inputPath: csvPath,
            outputPath: sqlitePath,
            tableName,
          })

          totalRows += result.totalRows
        }
      } finally {
        db.close()
      }
    }

    // Register dataset
    let datasetId: string | undefined

    if (!skipRegister && sqlitePath) {
      logStep({ step: 'register', message: 'Registering dataset...' }, ctx.output)

      const registry = new DatasetRegistry(join(dataDir, 'registry.db'))

      try {
        const checksum = await computeChecksum(sqlitePath)
        const fileInfo = Bun.file(sqlitePath)

        const dataset = registry.register({
          name,
          sourceType: 'kaggle',
          sourcePath: competition,
          sqlitePath,
          rowCount: totalRows,
          sizeBytes: fileInfo.size,
          checksum,
          metadata: {
            competition,
            csvFiles,
          },
        })

        datasetId = dataset.id

        progress({ step: 'registered', message: `Registered as ${name} v${dataset.version}` }, ctx.output)
      } finally {
        registry.close()
      }
    }

    return success({
      competition,
      kaggleDir,
      csvFiles,
      sqlitePath,
      totalRows,
      datasetId,
      name: skipRegister ? undefined : name,
    })
  },
}
