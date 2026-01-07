/**
 * Check Vertex AI job status
 */

import { z } from 'zod'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

// Simple info logging
function info(msg: string, _output: any): void {
  console.log(msg)
}

import { spawn } from 'child_process'

const StatusArgs = z.object({
  job: z.string().describe('Job name or ID'),
  region: z.string().default('us-central1').describe('GCP region'),
  watch: z.boolean().default(false).describe('Watch job until completion'),
})

export const status: CommandDefinition<typeof StatusArgs> = {
  name: 'vertex status',
  description: 'Check Vertex AI job status',
  help: `
Check the status of a Vertex AI training job.

Options:
  --job      Job name or resource ID (required)
  --region   GCP region (default: us-central1)
  --watch    Watch job until completion
`,
  examples: [
    'akk vertex status --job nllb-train-20251220T160000',
    'akk vertex status --job nllb-train-20251220T160000 --watch',
  ],
  args: StatusArgs,

  async run(args, ctx) {
    const config = ctx.config
    const project = config?.colab?.project || 'akkadian-481705'

    info(`Checking job status: ${args.job}`, ctx.output)

    try {
      // List jobs to find the one we want
      const result = await execCommand('gcloud', [
        'ai',
        'custom-jobs',
        'list',
        `--project=${project}`,
        `--region=${args.region}`,
        `--filter=displayName:${args.job}`,
        '--format=json',
      ])

      const jobs = JSON.parse(result || '[]')

      if (jobs.length === 0) {
        return error('JOB_NOT_FOUND', `Job not found: ${args.job}`)
      }

      const job = jobs[0]
      const state = job.state || 'UNKNOWN'
      const createTime = job.createTime ? new Date(job.createTime).toLocaleString() : 'Unknown'
      const endTime = job.endTime ? new Date(job.endTime).toLocaleString() : 'Running...'

      info(``, ctx.output)
      info(`Job:     ${job.displayName}`, ctx.output)
      info(`State:   ${state}`, ctx.output)
      info(`Created: ${createTime}`, ctx.output)
      info(`Ended:   ${endTime}`, ctx.output)

      if (job.error) {
        info(`Error:   ${job.error.message}`, ctx.output)
      }

      // If watching, poll until complete
      if (args.watch && !['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'].includes(state)) {
        info(``, ctx.output)
        info(`Watching job... (Ctrl+C to stop)`, ctx.output)

        while (true) {
          await new Promise((resolve) => setTimeout(resolve, 30000)) // Poll every 30s

          const updated = await execCommand('gcloud', [
            'ai',
            'custom-jobs',
            'list',
            `--project=${project}`,
            `--region=${args.region}`,
            `--filter=displayName:${args.job}`,
            '--format=json',
          ])

          const updatedJobs = JSON.parse(updated || '[]')
          if (updatedJobs.length === 0) break

          const updatedJob = updatedJobs[0]
          const newState = updatedJob.state

          if (newState !== state) {
            info(`  State changed: ${newState}`, ctx.output)
          }

          if (['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'].includes(newState)) {
            info(``, ctx.output)
            info(`Job completed with state: ${newState}`, ctx.output)
            break
          }
        }
      }

      return success({
        name: job.displayName,
        state: state,
        createTime: createTime,
        endTime: endTime,
        resourceName: job.name,
      })
    } catch (e) {
      return error('STATUS_FAILED', 'Failed to get job status', String(e))
    }
  },
}

async function execCommand(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''

    proc.stdout?.on('data', (data) => {
      stdout += data.toString()
    })
    proc.stderr?.on('data', (data) => {
      stderr += data.toString()
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        reject(new Error(stderr || `Command failed with code ${code}`))
      }
    })
  })
}
