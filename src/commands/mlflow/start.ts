/**
 * Start MLFlow tracking server
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { checkInstalled, checkServer, startServer, stopServer, listExperiments } from '../../lib/mlflow'

const StartArgs = z.object({
  port: z.number().default(5000).describe('Server port'),
  stop: z.boolean().default(false).describe('Stop the server'),
  open: z.boolean().default(true).describe('Open browser'),
  backend: z.string().optional().describe('Backend store URI'),
  artifacts: z.string().optional().describe('Artifact root location'),
})

export const start: CommandDefinition<typeof StartArgs> = {
  name: 'mlflow start',
  description: 'Start MLFlow tracking server',
  help: `
Starts the MLFlow tracking server for experiment management.
Uses sqlite backend and local artifact storage by default.

Options:
  --port      Server port (default: 5000)
  --stop      Stop the running server
  --open      Open browser after starting (default: true)
  --backend   Backend store URI (default: from akk.toml)
  --artifacts Artifact root location (default: from akk.toml)
`,
  examples: [
    'akk mlflow start',
    'akk mlflow start --port 5001',
    'akk mlflow start --stop',
    'akk mlflow start --backend sqlite:///mlflow.db --artifacts ./artifacts',
  ],
  args: StartArgs,

  async run(args, ctx) {
    const { port, stop, open, backend, artifacts } = args
    const config = ctx.config

    // Check MLFlow installed
    progress({ step: 'check', message: 'Checking MLFlow installation...' }, ctx.output)
    const installed = await checkInstalled()

    if (!installed.installed) {
      return error('MLFLOW_NOT_INSTALLED', 'MLFlow is not installed', 'Install with: pip install mlflow', {})
    }

    // Handle stop
    if (stop) {
      progress({ step: 'stop', message: 'Stopping MLFlow server...' }, ctx.output)
      const result = await stopServer(port)
      return success({
        action: 'stop',
        message: result.message,
      })
    }

    // Check if already running
    const running = await checkServer(port)
    if (running) {
      const experiments = await listExperiments(port)
      return success({
        action: 'already_running',
        port,
        url: `http://localhost:${port}`,
        experiments: experiments.map((e) => e.name),
        message: `MLFlow already running on port ${port}`,
      })
    }

    // Get config (CLI args take precedence)
    const trackingUri = backend || config?.mlflow?.tracking_uri || `sqlite:///${ctx.cwd}/mlflow/mlflow.db`
    const artifactLocation = artifacts || config?.mlflow?.artifact_location || `${ctx.cwd}/mlflow/artifacts`
    const serverPort = port // CLI arg takes precedence

    // Create directories
    const dbDir = trackingUri.replace('sqlite:///', '').replace('/mlflow.db', '')
    await Bun.spawn(['mkdir', '-p', dbDir]).exited
    await Bun.spawn(['mkdir', '-p', artifactLocation]).exited

    // Start server
    progress({ step: 'start', message: `Starting MLFlow server on port ${serverPort}...` }, ctx.output)

    const result = await startServer({
      trackingUri,
      artifactLocation,
      port: serverPort,
    })

    if (!result.success) {
      return error('START_FAILED', result.message, 'Check if port is in use or MLFlow is installed correctly', {
        port: serverPort,
        trackingUri,
      })
    }

    // Open browser
    if (open) {
      await Bun.spawn(['open', `http://localhost:${serverPort}`]).exited
    }

    // List experiments
    const experiments = await listExperiments(serverPort)

    return success({
      action: 'started',
      port: serverPort,
      url: `http://localhost:${serverPort}`,
      trackingUri,
      artifactLocation,
      experiments: experiments.map((e) => e.name),
      message: result.message,
    })
  },
}
