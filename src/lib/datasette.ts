/**
 * Datasette integration for data exploration
 */

import type { Subprocess } from 'bun'
import { commandExists, runCommand } from './process'

export interface DatasetteConfig {
  databases: string[]
  port: number
  host?: string
  openBrowser?: boolean
}

/**
 * Check if datasette is installed
 */
export async function checkDatasetteInstalled(): Promise<boolean> {
  return commandExists('datasette')
}

/**
 * Get datasette version
 */
export async function getDatasetteVersion(): Promise<string | null> {
  const result = await runCommand('datasette', ['--version'])

  if (!result.success) return null

  // Parse version from output like "datasette, version 0.64.5"
  const match = result.stdout.match(/version\s+([\d.]+)/)
  return match ? match[1] : null
}

/**
 * Start datasette server
 */
export async function startDatasette(config: DatasetteConfig): Promise<{
  process: Subprocess
  url: string
}> {
  const { databases, port, host = '127.0.0.1', openBrowser = true } = config

  // Build command args
  const args = ['datasette', 'serve', ...databases, '--port', String(port), '--host', host]

  // Add open flag if requested
  if (openBrowser) {
    args.push('--open')
  }

  // Start datasette
  const process = Bun.spawn(args, {
    stdout: 'inherit',
    stderr: 'inherit',
  })

  const url = `http://${host}:${port}`

  return { process, url }
}

/**
 * Open URL in browser (macOS)
 */
export async function openInBrowser(url: string): Promise<void> {
  await runCommand('open', [url])
}

/**
 * Check if port is in use
 */
export async function isPortInUse(port: number): Promise<boolean> {
  const result = await runCommand('lsof', ['-i', `:${port}`])
  return result.success
}

/**
 * Find an available port starting from the given port
 */
export async function findAvailablePort(startPort: number): Promise<number> {
  let port = startPort

  while (await isPortInUse(port)) {
    port++
    if (port > startPort + 100) {
      throw new Error(`Could not find available port in range ${startPort}-${port}`)
    }
  }

  return port
}
