/**
 * Doctor command - Check environment and dependencies
 */

import { z } from 'zod'
import type { CommandDefinition } from '../types/commands'
import { success, error } from '../lib/output'

const DoctorArgs = z.object({})

interface CheckResult {
  name: string
  status: 'ok' | 'warning' | 'error'
  version?: string
  message?: string
}

async function checkCommand(name: string, args: string[] = ['--version']): Promise<CheckResult> {
  try {
    const proc = Bun.spawn([name, ...args], {
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const output = await new Response(proc.stdout).text()
    const exitCode = await proc.exited

    if (exitCode === 0) {
      // Extract version from output
      const versionMatch = output.match(/(\d+\.\d+(?:\.\d+)?)/)?.[1]
      return {
        name,
        status: 'ok',
        version: versionMatch || 'installed',
      }
    }
    return {
      name,
      status: 'error',
      message: `Exit code: ${exitCode}`,
    }
  } catch (err) {
    return {
      name,
      status: 'error',
      message: 'Not found in PATH',
    }
  }
}

async function checkFile(name: string, path: string): Promise<CheckResult> {
  const file = Bun.file(path)
  const exists = await file.exists()

  return {
    name,
    status: exists ? 'ok' : 'warning',
    message: exists ? path : 'Not found',
  }
}

export const doctor: CommandDefinition<typeof DoctorArgs> = {
  name: 'doctor',
  description: 'Check environment and dependencies',
  examples: ['akk doctor'],
  args: DoctorArgs,

  async run(_args, ctx) {
    const checks: CheckResult[] = []

    // Check required tools
    checks.push(await checkCommand('kaggle'))
    checks.push(await checkCommand('gcloud'))
    checks.push(await checkCommand('gsutil'))
    checks.push(await checkCommand('mlflow'))
    checks.push(await checkCommand('jupytext'))
    checks.push(await checkCommand('python3'))

    // Check configuration files
    const homeDir = process.env.HOME || '~'
    checks.push(await checkFile('kaggle.json', `${homeDir}/.kaggle/kaggle.json`))

    // Check project config
    if (ctx.configPath) {
      checks.push({
        name: 'akk.toml',
        status: 'ok',
        message: ctx.configPath,
      })
    } else {
      checks.push({
        name: 'akk.toml',
        status: 'warning',
        message: 'Not found (using defaults)',
      })
    }

    // Count issues
    const errors = checks.filter((c) => c.status === 'error')
    const warnings = checks.filter((c) => c.status === 'warning')

    // Format results
    const results = checks.map((c) => {
      const icon = c.status === 'ok' ? '✓' : c.status === 'warning' ? '!' : '✗'
      const detail = c.version || c.message || ''
      return `${icon} ${c.name}: ${detail}`
    })

    if (errors.length > 0) {
      return error(
        'DOCTOR_FAILED',
        `${errors.length} required tool(s) missing`,
        'Install missing tools: kaggle, gcloud, gsutil, mlflow, jupytext',
        { checks }
      )
    }

    return success({
      summary: `${checks.length - warnings.length}/${checks.length} checks passed`,
      warnings: warnings.length,
      results,
    })
  },
}
