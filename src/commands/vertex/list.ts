/**
 * List Vertex AI jobs
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success, error } from '../../lib/output'

// Simple info logging
function info(msg: string, _output: any): void {
  console.log(msg)
}
import { spawn } from 'child_process'

const ListArgs = z.object({
  region: z.string().default('us-central1').describe('GCP region'),
  limit: z.number().default(10).describe('Number of jobs to show'),
  state: z.string().optional().describe('Filter by state (RUNNING, SUCCEEDED, FAILED)'),
})

export const list: CommandDefinition<typeof ListArgs> = {
  name: 'vertex list',
  description: 'List Vertex AI training jobs',
  help: `
List recent Vertex AI training jobs.

Options:
  --region   GCP region (default: us-central1)
  --limit    Number of jobs to show (default: 10)
  --state    Filter by state (RUNNING, SUCCEEDED, FAILED)
`,
  examples: [
    'akk vertex list',
    'akk vertex list --limit 20',
    'akk vertex list --state RUNNING',
  ],
  args: ListArgs,

  async run(args, ctx) {
    const config = ctx.config
    const project = config?.colab?.project || 'akkadian-481705'

    try {
      const gcloudArgs = [
        'ai', 'custom-jobs', 'list',
        `--project=${project}`,
        `--region=${args.region}`,
        `--limit=${args.limit}`,
        '--format=json',
      ]

      if (args.state) {
        gcloudArgs.push(`--filter=state:JOB_STATE_${args.state.toUpperCase()}`)
      }

      const result = await execCommand('gcloud', gcloudArgs)
      const jobs = JSON.parse(result || '[]')

      if (jobs.length === 0) {
        info(`No jobs found`, ctx.output)
        return success({ jobs: [] })
      }

      info(`Vertex AI Jobs (${jobs.length}):`, ctx.output)
      info(``, ctx.output)

      for (const job of jobs) {
        const state = (job.state || 'UNKNOWN').replace('JOB_STATE_', '')
        const name = job.displayName || 'Unnamed'
        const created = job.createTime ? new Date(job.createTime).toLocaleString() : 'Unknown'

        let stateIcon = '?'
        if (state === 'SUCCEEDED') stateIcon = '✓'
        else if (state === 'FAILED') stateIcon = '✗'
        else if (state === 'RUNNING') stateIcon = '→'
        else if (state === 'PENDING' || state === 'QUEUED') stateIcon = '○'

        info(`  ${stateIcon} ${name}`, ctx.output)
        info(`    State: ${state} | Created: ${created}`, ctx.output)
      }

      return success({
        jobs: jobs.map((j: any) => ({
          name: j.displayName,
          state: j.state,
          createTime: j.createTime,
          endTime: j.endTime,
        })),
      })
    } catch (e) {
      return error('LIST_FAILED', 'Failed to list jobs', String(e))
    }
  },
}

async function execCommand(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''

    proc.stdout?.on('data', (data) => { stdout += data.toString() })
    proc.stderr?.on('data', (data) => { stderr += data.toString() })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        reject(new Error(stderr || `Command failed with code ${code}`))
      }
    })
  })
}
