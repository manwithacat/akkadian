/**
 * Update prediction job status and metrics
 */

import { findCompetitionConfig } from '../../lib/config'
import { printError, printJson, printSuccess } from '../../lib/output'
import { formatPredictionJob, getPredictionJob, updatePredictionJobStatus } from '../../lib/prediction'
import type { CommandContext } from '../../types/commands'
import type { PredictionJobRecord, PredictionStatus } from '../../types/competition'

export interface UpdatePredictionOptions {
  id: string
  status?: string
  bleu?: number
  chrf?: number
  predictions?: string
  metrics?: string
  notes?: string
  json?: boolean
}

export async function update(options: UpdatePredictionOptions, _ctx: CommandContext): Promise<void> {
  if (!options.id) {
    printError('Prediction ID is required')
    return
  }

  // Find competition config
  const configPath = await findCompetitionConfig()
  if (!configPath) {
    printError("No competition.toml found. Run 'akk competition init' first.")
    return
  }

  // Validate status if provided
  if (options.status) {
    const validStatuses = ['queued', 'running', 'complete', 'error', 'cancelled']
    if (!validStatuses.includes(options.status)) {
      printError(`Invalid status: ${options.status}. Valid: ${validStatuses.join(', ')}`)
      return
    }
  }

  // Get current job
  const job = await getPredictionJob(options.id, configPath)
  if (!job) {
    printError(`Prediction job not found: ${options.id}`)
    return
  }

  // Build updates
  const updates: Partial<PredictionJobRecord> = {}

  // Update metrics
  if (options.bleu !== undefined || options.chrf !== undefined) {
    updates.metric_summary = {
      ...job.metric_summary,
      ...(options.bleu !== undefined ? { bleu: options.bleu } : {}),
      ...(options.chrf !== undefined ? { chrf: options.chrf } : {}),
    }
  }

  // Update artifacts
  if (options.predictions || options.metrics) {
    updates.artifacts = {
      predictions: options.predictions || job.artifacts?.predictions || '',
      metrics: options.metrics || job.artifacts?.metrics,
    }
  }

  // Update notes
  if (options.notes) {
    updates.notes = options.notes
  }

  try {
    const status = (options.status || job.status) as PredictionStatus
    await updatePredictionJobStatus(options.id, status, configPath, updates)

    // Fetch updated job
    const updatedJob = await getPredictionJob(options.id, configPath)

    if (options.json) {
      printJson(updatedJob)
    } else {
      printSuccess(`Updated prediction job: ${options.id}`)
      if (updatedJob) {
        console.log('\n' + formatPredictionJob(updatedJob))
      }
    }
  } catch (err) {
    printError(`Failed to update prediction: ${err instanceof Error ? err.message : String(err)}`)
  }
}
