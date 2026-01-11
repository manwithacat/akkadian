/**
 * List prediction jobs
 */

import { findCompetitionConfig } from '../../lib/config'
import { printError, printInfo, printJson } from '../../lib/output'
import { formatPredictionTable, type ListPredictionJobsOptions, listPredictionJobs } from '../../lib/prediction'
import type { CommandContext } from '../../types/commands'

export interface ListPredictionsOptions {
  status?: string
  model?: string
  dataset?: string
  limit?: number
  json?: boolean
}

export async function list(options: ListPredictionsOptions, _ctx: CommandContext): Promise<void> {
  // Find competition config
  const configPath = await findCompetitionConfig()
  if (!configPath) {
    printError("No competition.toml found. Run 'akk competition init' first.")
    return
  }

  // Build filter options
  const filterOptions: ListPredictionJobsOptions = {
    limit: options.limit,
  }

  if (options.status) {
    const validStatuses = ['queued', 'running', 'complete', 'error', 'cancelled']
    if (!validStatuses.includes(options.status)) {
      printError(`Invalid status: ${options.status}. Valid: ${validStatuses.join(', ')}`)
      return
    }
    filterOptions.status = options.status as ListPredictionJobsOptions['status']
  }

  if (options.model) {
    filterOptions.modelName = options.model
  }

  if (options.dataset) {
    filterOptions.dataset = options.dataset
  }

  // Fetch prediction jobs
  const jobs = await listPredictionJobs(configPath, filterOptions)

  if (options.json) {
    printJson(jobs)
    return
  }

  if (jobs.length === 0) {
    printInfo('No prediction jobs found.')
    printInfo('Register a prediction with: akk predict register --model <path> --dataset <name>')
    return
  }

  // Print table
  console.log(formatPredictionTable(jobs))
  console.log(`\nTotal: ${jobs.length} prediction job(s)`)
}
