/**
 * Launch Datasette to explore dataset files
 */

import { join } from 'path'
import { z } from 'zod'
import { DatasetRegistry } from '../../lib/data-registry'
import { checkDatasetteInstalled, findAvailablePort, isPortInUse, startDatasette } from '../../lib/datasette'
import { error, logStep, progress, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import { parseDatasetRef } from '../../types/data'

const ExploreArgs = z.object({
  name: z.string().optional().describe('Dataset name to explore (default: all)'),
  version: z.number().optional().describe('Specific version (default: latest)'),
  port: z.number().default(8001).describe('Port for Datasette server'),
  noBrowser: z.boolean().default(false).describe('Do not open browser automatically'),
})

export const explore: CommandDefinition<typeof ExploreArgs> = {
  name: 'data explore',
  description: 'Launch Datasette to explore dataset files',
  help: `
Launches a Datasette server to explore registered datasets.

By default, opens all registered SQLite databases. Use --name to focus on
a specific dataset, optionally with --version for a specific version.

Options:
  --name        Dataset name to explore
  --version     Specific version number (requires --name)
  --port        Port for Datasette server (default: 8001)
  --no-browser  Don't open browser automatically

Requires datasette to be installed: pip install datasette
`,
  examples: [
    'akk data explore',
    'akk data explore --name raw',
    'akk data explore --name v2_augmented --version 1',
    'akk data explore --port 8002 --no-browser',
  ],
  args: ExploreArgs,

  async run(args, ctx) {
    const { name, version, port: requestedPort, noBrowser } = args

    // Check datasette is installed
    const installed = await checkDatasetteInstalled()
    if (!installed) {
      return error('DATASETTE_NOT_INSTALLED', 'Datasette is not installed', 'Install with: pip install datasette', {})
    }

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
      // Collect database files to serve
      const databases: string[] = []

      if (name) {
        // Get specific dataset
        const dataset = version ? registry.getVersion(name, version) : registry.getLatestVersion(name)

        if (!dataset) {
          const ref = version ? `${name}:${version}` : name
          return error(
            'DATASET_NOT_FOUND',
            `Dataset not found: ${ref}`,
            'Run "akk data list" to see available datasets',
            { name, version }
          )
        }

        // Check file exists
        const dbFile = Bun.file(dataset.sqlitePath)
        if (!(await dbFile.exists())) {
          return error(
            'FILE_NOT_FOUND',
            `Database file not found: ${dataset.sqlitePath}`,
            'The dataset may have been moved or deleted',
            { path: dataset.sqlitePath }
          )
        }

        databases.push(dataset.sqlitePath)
        logStep({ step: 'found', message: `Opening ${dataset.name}:${dataset.version}` }, ctx.output)
      } else {
        // Get all unique datasets (latest versions)
        const names = registry.getDatasetNames()

        if (names.length === 0) {
          return error(
            'NO_DATASETS',
            'No datasets registered',
            'Run "akk data download" or "akk data register" first',
            {}
          )
        }

        // Collect latest version of each dataset
        for (const dsName of names) {
          const dataset = registry.getLatestVersion(dsName)
          if (dataset) {
            const dbFile = Bun.file(dataset.sqlitePath)
            if (await dbFile.exists()) {
              databases.push(dataset.sqlitePath)
            }
          }
        }

        logStep({ step: 'found', message: `Opening ${databases.length} datasets` }, ctx.output)
      }

      // Also include registry for metadata exploration
      databases.push(registryPath)

      // Find available port
      let port = requestedPort
      if (await isPortInUse(port)) {
        logStep({ step: 'port', message: `Port ${port} in use, finding alternative...` }, ctx.output)
        port = await findAvailablePort(requestedPort + 1)
      }

      // Start datasette
      logStep({ step: 'start', message: `Starting Datasette on port ${port}...` }, ctx.output)

      const { process: dsProcess, url } = await startDatasette({
        databases,
        port,
        openBrowser: !noBrowser,
      })

      logStep({ step: 'ready', message: `Datasette running at ${url}` }, ctx.output)

      // Wait for process (blocks until terminated)
      const exitCode = await dsProcess.exited

      return success({
        url,
        port,
        databases: databases.length,
        exitCode,
      })
    } finally {
      registry.close()
    }
  },
}
